"""Joint distribution analyzer + inverse-frequency sampler.

Analyzes a host YOLO label directory and produces:
  - 4D histogram H[class, scale_bin, cx_bin, cy_bin] of training instances
  - per-class scale quartiles (scale = sqrt(w_norm * h_norm))
  - per-class linear regression cy_norm -> h_norm (used as a depth-free perspective prior)

Sampler draws (class, target_scale_norm, target_cx_norm, target_cy_norm) with weights
proportional to (1 / (hist + floor)) ** (1 / temperature), so tail bins are oversampled.

CLI:
  python -m xpaste.aug.distribution --labels_dir <path>/train/labels --out cache/dist.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS

N_SCALE_BINS = 4
N_CX_BINS = 4
N_CY_BINS = 4


@dataclass
class JointHist:
    counts: np.ndarray                            # shape [4, 4, 4, 4]
    scale_quartiles: dict[str, list[float]]       # per-class [q25, q50, q75]
    cy_regression: dict[str, tuple[float, float, float]]  # per-class (a, b, R^2): h = a + b * cy
    n_train_imgs: int
    n_instances_total: int
    per_class_count: dict[str, int]


def parse_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            continue
        if cls < 0 or cls >= len(CLASSES):
            continue
        out.append((cls, cx, cy, w, h))
    return out


def compute_joint_histogram(labels_dir: Path, exclude_files: set[str] | None = None) -> JointHist:
    labels_dir = Path(labels_dir)
    exclude_files = exclude_files or set()

    per_class_scales: dict[str, list[float]] = {c: [] for c in CLASSES}
    per_class_cy_h: dict[str, list[tuple[float, float]]] = {c: [] for c in CLASSES}
    rows: list[tuple[int, float, float, float]] = []
    img_count = 0

    for txt in sorted(labels_dir.glob("*.txt")):
        img_count += 1
        if txt.name in exclude_files:
            continue
        for cls, cx, cy, w, h in parse_yolo_label(txt):
            scale = float(np.sqrt(max(1e-12, w * h)))
            per_class_scales[CLASSES[cls]].append(scale)
            per_class_cy_h[CLASSES[cls]].append((cy, h))
            rows.append((cls, scale, cx, cy))

    quartiles: dict[str, list[float]] = {}
    for c in CLASSES:
        s = np.asarray(per_class_scales[c], dtype=np.float64)
        if s.size < 4:
            quartiles[c] = [0.05, 0.10, 0.25]
        else:
            quartiles[c] = [
                float(np.quantile(s, 0.25)),
                float(np.quantile(s, 0.50)),
                float(np.quantile(s, 0.75)),
            ]

    regression: dict[str, tuple[float, float, float]] = {}
    for c in CLASSES:
        pairs = per_class_cy_h[c]
        if len(pairs) < 10:
            regression[c] = (0.0, 0.0, 0.0)
            continue
        cy_arr = np.asarray([p[0] for p in pairs])
        h_arr = np.asarray([p[1] for p in pairs])
        b, a = np.polyfit(cy_arr, h_arr, 1)
        h_pred = a + b * cy_arr
        ss_res = float(np.sum((h_arr - h_pred) ** 2))
        ss_tot = float(np.sum((h_arr - h_arr.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        regression[c] = (float(a), float(b), float(r2))

    counts = np.zeros((len(CLASSES), N_SCALE_BINS, N_CX_BINS, N_CY_BINS), dtype=np.int64)
    for cls, scale, cx, cy in rows:
        cname = CLASSES[cls]
        sb = _scale_bin(scale, quartiles[cname])
        cxb = int(np.clip(cx * N_CX_BINS, 0, N_CX_BINS - 1))
        cyb = int(np.clip(cy * N_CY_BINS, 0, N_CY_BINS - 1))
        counts[cls, sb, cxb, cyb] += 1

    return JointHist(
        counts=counts,
        scale_quartiles=quartiles,
        cy_regression=regression,
        n_train_imgs=img_count,
        n_instances_total=int(counts.sum()),
        per_class_count={c: int(counts[CLASS_TO_IDX[c]].sum()) for c in CLASSES},
    )


def _scale_bin(scale: float, quartiles: list[float]) -> int:
    q25, q50, q75 = quartiles
    if scale < q25:
        return 0
    if scale < q50:
        return 1
    if scale < q75:
        return 2
    return 3


def scale_bin_bounds(quartiles: list[float], bin_idx: int) -> tuple[float, float]:
    """Return [lo, hi] for a scale bin (used to draw a target scale uniformly)."""
    q25, q50, q75 = quartiles
    if bin_idx == 0:
        return max(1e-3, q25 * 0.5), q25
    if bin_idx == 1:
        return q25, q50
    if bin_idx == 2:
        return q50, q75
    return q75, min(1.0, q75 * 1.6)


class InverseFreqSampler:
    """Draw (class, target_scale_norm, target_cx_norm, target_cy_norm) biased toward tail bins.

    weights[c, s, x, y] = (1 / (counts[c, s, x, y] + floor)) ** (1 / temperature)
    Probabilities are normalized over all 256 cells (4 cls x 4 x 4 x 4).
    """

    def __init__(self, hist: JointHist, temperature: float = 1.0, floor: int = 1):
        self.hist = hist
        self.temperature = float(temperature)
        self.floor = int(floor)
        w = 1.0 / (hist.counts.astype(np.float64) + self.floor)
        w = w ** (1.0 / max(1e-3, self.temperature))
        w = w / w.sum()
        self.flat_weights = w.flatten()
        self.shape = w.shape

    def sample(self, rng: np.random.Generator) -> tuple[str, float, float, float]:
        flat_idx = int(rng.choice(self.flat_weights.size, p=self.flat_weights))
        c, sb, cxb, cyb = np.unravel_index(flat_idx, self.shape)
        cname = IDX_TO_CLASS[int(c)]
        s_lo, s_hi = scale_bin_bounds(self.hist.scale_quartiles[cname], int(sb))
        target_scale = float(rng.uniform(s_lo, s_hi))
        target_cx = float(rng.uniform(cxb / N_CX_BINS, (cxb + 1) / N_CX_BINS))
        target_cy = float(rng.uniform(cyb / N_CY_BINS, (cyb + 1) / N_CY_BINS))
        return cname, target_scale, target_cx, target_cy


def save_hist(hist: JointHist, path: Path) -> None:
    payload = {
        "counts": hist.counts.tolist(),
        "scale_quartiles": hist.scale_quartiles,
        "cy_regression": {c: list(v) for c, v in hist.cy_regression.items()},
        "n_train_imgs": hist.n_train_imgs,
        "n_instances_total": hist.n_instances_total,
        "per_class_count": hist.per_class_count,
        "classes": list(CLASSES),
        "shape": list(hist.counts.shape),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def load_hist(path: Path) -> JointHist:
    p = json.loads(Path(path).read_text())
    return JointHist(
        counts=np.asarray(p["counts"], dtype=np.int64),
        scale_quartiles=p["scale_quartiles"],
        cy_regression={c: tuple(v) for c, v in p["cy_regression"].items()},
        n_train_imgs=p["n_train_imgs"],
        n_instances_total=p["n_instances_total"],
        per_class_count=p["per_class_count"],
    )


def print_summary(hist: JointHist) -> None:
    print(f"images: {hist.n_train_imgs}, instances: {hist.n_instances_total}")
    for c in CLASSES:
        q = hist.scale_quartiles[c]
        a, b, r2 = hist.cy_regression[c]
        n = hist.per_class_count[c]
        print(
            f"  {c:18s} n={n:5d}  scale q25/50/75={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}  "
            f"cy->h: a={a:.3f} b={b:+.3f} R2={r2:.3f}"
        )
    print("\n  per-class cell counts (rows=scale bin, cols=cy bin, summed over cx):")
    for c in CLASSES:
        ci = CLASS_TO_IDX[c]
        m = hist.counts[ci].sum(axis=1)  # [scale, cy]
        print(f"    {c}:")
        for s in range(N_SCALE_BINS):
            print("      s=" + str(s) + " | " + " ".join(f"{m[s, y]:4d}" for y in range(N_CY_BINS)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels_dir", required=True, help="train labels directory (YOLO .txt)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--exclude_list", default=None, help="optional file with one .txt name per line to exclude")
    args = ap.parse_args()

    exclude = set()
    if args.exclude_list:
        for line in Path(args.exclude_list).read_text().splitlines():
            line = line.strip()
            if line:
                exclude.add(line)

    hist = compute_joint_histogram(Path(args.labels_dir), exclude_files=exclude)
    save_hist(hist, Path(args.out))
    print_summary(hist)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
