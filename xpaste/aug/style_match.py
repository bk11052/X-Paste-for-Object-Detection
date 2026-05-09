"""Style-matched instance selection (Lab histogram chi-square) + post-paste L-channel match.

Index step (one-shot, cached to .npz):
  For each RGBA instance crop in the pool, compute:
    - 8x8x8 Lab histogram (alpha-masked)
    - aspect (w / h)
    - height in px
    - source class (from parent directory name or filename prefix)

Selection step (per paste):
  Given (host_crop_rgb, target_class, target_aspect, target_height_px):
    1. Filter pool of target_class to candidates with aspect within +-30% AND
       height >= 0.5 * target_height_px (avoid heavy upscaling).
    2. Compute chi-square distance between host_crop Lab histogram and each candidate.
    3. Sample one of top-k closest (k=8 by default) for diversity.

Post-paste step:
  After alpha-blend, optionally adjust the pasted pixels' L channel toward the host
  ring (margin around bbox) using a clipped histogram match (max L shift = 20 levels).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class InstanceRecord:
    path: str
    cls: str
    pose_slug: str
    width_px: int
    height_px: int
    aspect: float
    lab_hist: np.ndarray   # flat 512


@dataclass
class PoolIndex:
    by_class: dict[str, list[InstanceRecord]] = field(default_factory=dict)


def rgb_to_lab_bins(rgb_uint8: np.ndarray, mask: np.ndarray | None = None,
                    bins: tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    """Compute a flattened Lab histogram. rgb_uint8: HxWx3 uint8.

    Uses skimage if available, otherwise a quick BGR-LMN proxy.
    """
    try:
        from skimage.color import rgb2lab
        lab = rgb2lab(rgb_uint8.astype(np.float32) / 255.0)
        L = lab[..., 0] / 100.0
        a = (lab[..., 1] + 128.0) / 255.0
        b = (lab[..., 2] + 128.0) / 255.0
    except Exception:
        # Fallback: simple normalized RGB (still a valid 3D histogram, just not Lab).
        rgb = rgb_uint8.astype(np.float32) / 255.0
        L = rgb.mean(axis=-1)
        a = (rgb[..., 0] - rgb[..., 1] + 1.0) / 2.0
        b = (rgb[..., 2] - rgb[..., 1] + 1.0) / 2.0

    if mask is not None:
        sel = mask.astype(bool)
        L = L[sel]; a = a[sel]; b = b[sel]
    else:
        L = L.flatten(); a = a.flatten(); b = b.flatten()
    if L.size == 0:
        return np.zeros(bins[0] * bins[1] * bins[2], dtype=np.float32)
    Lb = np.clip((L * bins[0]).astype(np.int32), 0, bins[0] - 1)
    ab_ = np.clip((a * bins[1]).astype(np.int32), 0, bins[1] - 1)
    bb_ = np.clip((b * bins[2]).astype(np.int32), 0, bins[2] - 1)
    flat = Lb * (bins[1] * bins[2]) + ab_ * bins[2] + bb_
    h = np.bincount(flat, minlength=bins[0] * bins[1] * bins[2]).astype(np.float32)
    h /= max(1.0, h.sum())
    return h


def hist_chi2(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-7
    return float(0.5 * np.sum(((a - b) ** 2) / (a + b + eps)))


def _class_from_path(p: Path) -> tuple[str, str]:
    """Infer (class, pose_slug) from path. Pool layout from segment_pose_hf.py:
        pool_dir/<category>__<pose_slug>/<idx>.png
    """
    parent = p.parent.name
    if "__" in parent:
        cls, slug = parent.split("__", 1)
        return cls, slug
    return parent, parent


def index_pool(pool_dir: Path, exts: tuple[str, ...] = (".png",)) -> PoolIndex:
    pool_dir = Path(pool_dir)
    by_class: dict[str, list[InstanceRecord]] = {}
    files = [p for ext in exts for p in pool_dir.rglob(f"*{ext}")]
    for p in files:
        try:
            img = Image.open(p)
        except Exception:
            continue
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        arr = np.asarray(img)
        if arr.shape[-1] != 4:
            continue
        rgb = arr[..., :3]
        alpha = arr[..., 3]
        mask = alpha > 32
        if mask.sum() < 100:
            continue
        h_px, w_px = arr.shape[:2]
        # Tighten box to non-zero alpha for aspect
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max() + 1)
        x1, x2 = int(xs.min()), int(xs.max() + 1)
        bh = max(1, y2 - y1); bw = max(1, x2 - x1)
        cls, slug = _class_from_path(p)
        rec = InstanceRecord(
            path=str(p), cls=cls, pose_slug=slug,
            width_px=bw, height_px=bh,
            aspect=float(bw) / float(bh),
            lab_hist=rgb_to_lab_bins(rgb, mask=mask),
        )
        by_class.setdefault(cls, []).append(rec)
    return PoolIndex(by_class=by_class)


def save_index(index: PoolIndex, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for cls, recs in index.by_class.items():
        payload[f"{cls}__path"] = np.asarray([r.path for r in recs])
        payload[f"{cls}__pose"] = np.asarray([r.pose_slug for r in recs])
        payload[f"{cls}__w"] = np.asarray([r.width_px for r in recs], dtype=np.int32)
        payload[f"{cls}__h"] = np.asarray([r.height_px for r in recs], dtype=np.int32)
        payload[f"{cls}__a"] = np.asarray([r.aspect for r in recs], dtype=np.float32)
        payload[f"{cls}__hist"] = np.stack([r.lab_hist for r in recs]) if recs else np.zeros((0, 512), dtype=np.float32)
    payload["__classes__"] = np.asarray(list(index.by_class.keys()))
    np.savez_compressed(path, **payload)


def load_index(path: Path) -> PoolIndex:
    data = np.load(path, allow_pickle=False)
    classes = list(data["__classes__"])
    by_class: dict[str, list[InstanceRecord]] = {}
    for cls in classes:
        paths = data[f"{cls}__path"]
        poses = data[f"{cls}__pose"]
        ws = data[f"{cls}__w"]; hs = data[f"{cls}__h"]; aspects = data[f"{cls}__a"]
        hists = data[f"{cls}__hist"]
        recs = [
            InstanceRecord(
                path=str(paths[i]), cls=str(cls), pose_slug=str(poses[i]),
                width_px=int(ws[i]), height_px=int(hs[i]),
                aspect=float(aspects[i]), lab_hist=hists[i].astype(np.float32),
            )
            for i in range(len(paths))
        ]
        by_class[str(cls)] = recs
    return PoolIndex(by_class=by_class)


def select_topk_match(
    host_crop_rgb: np.ndarray,
    pool: PoolIndex,
    cls: str,
    target_aspect: float,
    target_h_px: int,
    rng: np.random.Generator,
    k: int = 8,
    aspect_tol: float = 0.30,
) -> InstanceRecord | None:
    pool_recs = pool.by_class.get(cls, [])
    if not pool_recs:
        return None
    # filter by aspect and height
    cands = [
        r for r in pool_recs
        if (1.0 - aspect_tol) * target_aspect <= r.aspect <= (1.0 + aspect_tol) * target_aspect
        and r.height_px >= 0.5 * target_h_px
    ]
    if len(cands) < k:
        cands = pool_recs
    host_hist = rgb_to_lab_bins(host_crop_rgb)
    dists = np.asarray([hist_chi2(host_hist, r.lab_hist) for r in cands])
    order = np.argsort(dists)
    top = order[: min(k, len(cands))]
    pick = int(rng.choice(top))
    return cands[pick]


def lab_histogram_match_local(
    composed_rgb: np.ndarray,
    paste_alpha: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    margin_px: int = 8,
    max_L_shift: float = 20.0,
    ab_weight: float = 0.4,
) -> np.ndarray:
    """Adjust the L (and softly a, b) channels of pasted pixels toward the host ring stats.

    composed_rgb : HxWx3 uint8 (already pasted)
    paste_alpha  : HxWx float in [0, 1] (only pixels > 0 are adjusted)
    bbox_xyxy    : the paste bbox; the host ring is the margin around it
    Returns a new HxWx3 uint8 with the matched region.
    """
    try:
        from skimage.color import rgb2lab, lab2rgb
    except Exception:
        return composed_rgb  # silently skip if skimage missing

    H, W = composed_rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    rx1 = max(0, x1 - margin_px); rx2 = min(W, x2 + margin_px)
    ry1 = max(0, y1 - margin_px); ry2 = min(H, y2 + margin_px)
    ring_mask = np.zeros((H, W), dtype=bool)
    ring_mask[ry1:ry2, rx1:rx2] = True
    ring_mask[y1:y2, x1:x2] = False  # exclude the paste itself
    ring_mask &= paste_alpha < 1e-3   # also exclude any nearby paste pixels
    if ring_mask.sum() < 50:
        return composed_rgb

    src_mask = paste_alpha > 0.05
    if src_mask.sum() < 50:
        return composed_rgb

    lab = rgb2lab(composed_rgb.astype(np.float32) / 255.0)
    L_src = lab[..., 0][src_mask]
    L_ring = lab[..., 0][ring_mask]
    a_src = lab[..., 1][src_mask]
    a_ring = lab[..., 1][ring_mask]
    b_src = lab[..., 2][src_mask]
    b_ring = lab[..., 2][ring_mask]

    dL = float(np.median(L_ring) - np.median(L_src))
    dL = max(-max_L_shift, min(max_L_shift, dL))
    da = ab_weight * (float(np.median(a_ring)) - float(np.median(a_src)))
    db = ab_weight * (float(np.median(b_ring)) - float(np.median(b_src)))

    lab[..., 0][src_mask] = np.clip(lab[..., 0][src_mask] + dL, 0, 100)
    lab[..., 1][src_mask] = np.clip(lab[..., 1][src_mask] + da, -128, 127)
    lab[..., 2][src_mask] = np.clip(lab[..., 2][src_mask] + db, -128, 127)

    rgb_new = (np.clip(lab2rgb(lab), 0, 1) * 255.0).astype(np.uint8)
    return rgb_new


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool_dir", required=True)
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()
    idx = index_pool(Path(args.pool_dir))
    save_index(idx, Path(args.out))
    print(f"indexed pool from {args.pool_dir}")
    for c, recs in idx.by_class.items():
        print(f"  {c}: {len(recs)} instances")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
