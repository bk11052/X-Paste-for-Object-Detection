"""
Capture per-stage artifacts of the REAL xpaste/aug pipeline for ONE host image,
for the single-example "pipeline tour" visualization video.

Two modes:
  (A) --from_meta meta.jsonl --host <file>   reproduce EXACTLY the paste that
      build_augmented.py produced for that image (recommended: pick a good image
      from an aug_E run, then visualize that very one).
  (B) auto: sample a fresh paste for --host (or auto-pick a host) using the real
      InverseFreqSampler + style-match.

Outputs (to --out, default ./demo_stages): host.png, mask_paste.png,
hist_inset.png, hist_target.png, instance_src.png, instance_placed.png,
corrected.png, meta.json   (consumed by render_example_tour.py)

Example (server, GPU):
  python tools/viz/capture_stages.py \
    --host_root data/military_v1/real/train \
    --host  some_host.jpg \
    --from_meta data/military_v1/aug_E/train/meta.jsonl \
    --pool_index cache/pool_index_v1.npz --hist cache/dist_v1.json \
    --out runs/viz/demo_stages --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# repo root = two levels up from this file (tools/viz/..)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from xpaste.aug import CLASSES, CLASS_TO_IDX                                    # noqa: E402
from xpaste.aug.distribution import (                                          # noqa: E402
    compute_joint_histogram, InverseFreqSampler, _scale_bin,
    N_CX_BINS, N_CY_BINS, N_SCALE_BINS, load_hist,
)
from xpaste.aug.host_scene import HostSceneAnalyzer, yolo_to_xyxy             # noqa: E402
from xpaste.aug.scale_heuristic import choose_scale, CLASS_ASPECT            # noqa: E402
from xpaste.aug.style_match import (                                          # noqa: E402
    index_pool, load_index, select_topk_match, rgb_to_lab_bins, hist_chi2,
    lab_histogram_match_local,
)
from xpaste.aug.build_augmented import _plan_bbox, _alpha_paste              # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png")


def feather_rgba(crop_rgb, feather=10):
    h, w = crop_rgb.shape[:2]
    a = np.ones((h, w), np.float32)
    for i in range(feather):
        v = (i + 1) / (feather + 1)
        a[i, :] = np.minimum(a[i, :], v); a[h - 1 - i, :] = np.minimum(a[h - 1 - i, :], v)
        a[:, i] = np.minimum(a[:, i], v); a[:, w - 1 - i] = np.minimum(a[:, w - 1 - i], v)
    return np.dstack([crop_rgb, (a * 255).astype(np.uint8)])


def build_demo_pool(host_root: Path, pool_dir: Path, per_class=40, min_px=48):
    """Fallback pool from real GT crops when no SD pool is available."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    counts = {c: 0 for c in CLASSES}
    imgs = sorted(p for p in (host_root / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    rng = random.Random(0); rng.shuffle(imgs)
    for ip in imgs:
        if all(v >= per_class for v in counts.values()):
            break
        lp = host_root / "labels" / (ip.stem + ".txt")
        if not lp.exists():
            continue
        try:
            im = Image.open(ip).convert("RGB")
        except Exception:
            continue
        boxes, names = yolo_to_xyxy(lp, *im.size)
        for (x1, y1, x2, y2), cls in zip(boxes, names):
            if counts[cls] >= per_class or (x2 - x1) < min_px or (y2 - y1) < min_px:
                continue
            d = pool_dir / f"{cls}__gt"; d.mkdir(exist_ok=True)
            Image.fromarray(feather_rgba(np.asarray(im.crop((x1, y1, x2, y2))))).save(
                d / f"{counts[cls]:03d}.png")
            counts[cls] += 1
    print("demo pool counts:", counts)


def pick_host(analyzer, host_root: Path, max_probe=14):
    lbl_dir = host_root / "labels"; img_dir = host_root / "images"
    cands = []
    for lp in sorted(lbl_dir.glob("*.txt")):
        ip = next((img_dir / (lp.stem + e) for e in IMG_EXTS if (img_dir / (lp.stem + e)).exists()), None)
        if ip is None:
            continue
        rows = [r.split() for r in lp.read_text().splitlines() if len(r.split()) >= 5]
        if not (1 <= len(rows) <= 2):
            continue
        try:
            cys = [float(r[2]) for r in rows]; areas = [float(r[3]) * float(r[4]) for r in rows]
        except ValueError:
            continue
        score = sum(cy > 0.5 for cy in cys) - 3 * sum(a > 0.25 for a in areas)
        cands.append((score, ip, lp))
    cands.sort(key=lambda t: -t[0])
    print(f"{len(cands)} candidates; SegFormer-probing top {max_probe}")
    for _, ip, lp in cands[:max_probe]:
        host = analyzer.analyze(ip, lp)
        gm = host.region_masks.get("ground", np.zeros((host.H, host.W), bool))
        rm = host.region_masks.get("road", np.zeros((host.H, host.W), bool))
        frac = float((gm | rm).sum()) / (host.H * host.W)
        print(f"  {ip.name}: accept={host.accept} region={frac:.2f}")
        if host.accept and frac >= 0.15:
            return ip, lp, host
    for _, ip, lp in cands[:max_probe]:
        host = analyzer.analyze(ip, lp)
        if host.accept:
            return ip, lp, host
    raise SystemExit("no acceptable host found")


def hist_heatmap(hist, cls, cell, path, highlight):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ci = CLASS_TO_IDX[cls]
    m = hist.counts[ci].sum(axis=1)
    fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=140)
    ax.imshow(m, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xlabel("cy bin"); ax.set_ylabel("scale bin")
    ax.set_title(f"{cls}  joint hist", fontsize=10)
    ax.set_xticks(range(N_CY_BINS)); ax.set_yticks(range(N_SCALE_BINS))
    if highlight:
        _, sb, cxb, cyb = cell
        ax.add_patch(plt.Rectangle((cyb - 0.5, sb - 0.5), 1, 1, fill=False, edgecolor="#f4a000", lw=4))
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)


def find_meta_record(meta_path: Path, host_file: str):
    for line in Path(meta_path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("src_img") == host_file or rec.get("img") == host_file:
            if rec.get("pastes"):
                return rec
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host_root", required=True, help="dir with images/ and labels/")
    ap.add_argument("--host", default=None, help="specific host image filename (else auto-pick)")
    ap.add_argument("--from_meta", default=None, help="meta.jsonl from build_augmented; reproduce exact paste")
    ap.add_argument("--pool_index", default=None)
    ap.add_argument("--pool_dir", default=None)
    ap.add_argument("--hist", default=None, help="cached dist json (else computed from host_root/labels)")
    ap.add_argument("--out", default="demo_stages")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None, help="e.g. cuda:0 / cuda:1 / cpu")
    ap.add_argument("--seg_model", default="nvidia/segformer-b5-finetuned-ade-640-640")
    ap.add_argument("--demo_pool", action="store_true", help="build GT-crop pool if no SD pool given")
    ap.add_argument("--prefer_cls", default="military_vehicle")
    args = ap.parse_args()

    host_root = Path(args.host_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # pool
    if args.pool_index and Path(args.pool_index).exists():
        pool = load_index(Path(args.pool_index))
    elif args.pool_dir:
        pool = index_pool(Path(args.pool_dir))
    elif args.demo_pool:
        pd = out / "_demo_pool"
        if not pd.exists() or not any(pd.iterdir()):
            build_demo_pool(host_root, pd)
        pool = index_pool(pd)
    else:
        raise SystemExit("provide --pool_index or --pool_dir or --demo_pool")
    print("pool:", {c: len(v) for c, v in pool.by_class.items()})

    print(f"loading SegFormer ({args.seg_model}) on {args.device or 'auto'} ...")
    analyzer = HostSceneAnalyzer(seg_model=args.seg_model, device=args.device)

    # host
    if args.host:
        ip = next((host_root / "images" / args.host for _ in [0]
                   if (host_root / "images" / args.host).exists()), None)
        if ip is None:
            raise SystemExit(f"host not found: {args.host}")
        lp = host_root / "labels" / (ip.stem + ".txt")
        host = analyzer.analyze(ip, lp)
    else:
        ip, lp, host = pick_host(analyzer, host_root)
    print(f"HOST: {ip.name} ({host.W}x{host.H}) accept={host.accept} gt={host.gt_classes}")

    host_np = np.asarray(host.image.convert("RGB"))
    Image.fromarray(host_np).save(out / "host.png")
    gm = host.region_masks.get("ground", np.zeros((host.H, host.W), bool))
    rm = host.region_masks.get("road", np.zeros((host.H, host.W), bool))
    Image.fromarray(((gm | rm).astype(np.uint8) * 255)).save(out / "mask_paste.png")

    hist = load_hist(Path(args.hist)) if args.hist else compute_joint_histogram(host_root / "labels")

    # decide paste: from_meta (exact) or sample
    if args.from_meta:
        rec = find_meta_record(Path(args.from_meta), ip.name)
        if rec is None:
            raise SystemExit(f"no paste record for {ip.name} in {args.from_meta}")
        p0 = rec["pastes"][0]
        cls = p0["cls"]; bbox = tuple(int(v) for v in p0["bbox_xyxy"])
        ts = p0.get("target_scale", 0.1); tcx = p0.get("target_cx", 0.5); tcy = p0.get("target_cy", 0.7)
        method = p0.get("scale_method", "gt_anchor")
        inst_path = p0["instance_src"]
        print(f"[from_meta] cls={cls} bbox={bbox} inst={Path(inst_path).name}")
    else:
        rng = np.random.default_rng(args.seed)
        sampler = InverseFreqSampler(hist, temperature=args.temperature)
        chosen = None
        for tnum in range(400):
            cls, ts, tcx, tcy = sampler.sample(rng)
            if not pool.by_class.get(cls):
                continue
            ta = CLASS_ASPECT.get(cls, 1.0)
            sr = choose_scale(cls=cls, target_scale_norm=ts, target_cy_norm=tcy, H_img=host.H,
                              gt_boxes_xyxy=host.gt_boxes_xyxy, gt_classes=host.gt_classes,
                              cy_regression=hist.cy_regression, target_aspect=ta)
            bb = _plan_bbox(host, cls, tcx, tcy, sr.height_px, sr.width_px,
                            existing=list(host.gt_boxes_xyxy), rng=rng, max_attempts=40)
            if bb is None:
                continue
            if chosen is None:
                chosen = (cls, ts, tcx, tcy, sr, bb)
            if tnum < 200 and cls != args.prefer_cls:
                continue
            chosen = (cls, ts, tcx, tcy, sr, bb); break
        if chosen is None:
            raise SystemExit("could not plan any paste")
        cls, ts, tcx, tcy, sr, bbox = chosen
        method = sr.method
        ta = CLASS_ASPECT.get(cls, 1.0)
        rng2 = np.random.default_rng(args.seed + 1)
        rec_inst = select_topk_match(host_np[bbox[1]:bbox[3], bbox[0]:bbox[2]], pool, cls, ta,
                                     sr.height_px, rng2, k=8)
        if rec_inst is None:
            raise SystemExit("style match returned None")
        inst_path = rec_inst.path

    x1, y1, x2, y2 = bbox
    print(f"target cls={cls} scale_method={method} bbox={bbox}")

    sb = _scale_bin(ts, hist.scale_quartiles[cls])
    cxb = int(np.clip(tcx * N_CX_BINS, 0, N_CX_BINS - 1))
    cyb = int(np.clip(tcy * N_CY_BINS, 0, N_CY_BINS - 1))
    cell = (CLASS_TO_IDX[cls], sb, cxb, cyb)
    hist_heatmap(hist, cls, cell, out / "hist_inset.png", False)
    hist_heatmap(hist, cls, cell, out / "hist_target.png", True)

    inst_rgba = np.asarray(Image.open(inst_path).convert("RGBA"))
    Image.fromarray(inst_rgba).save(out / "instance_src.png")
    chi2 = hist_chi2(rgb_to_lab_bins(host_np[y1:y2, x1:x2]), rgb_to_lab_bins(inst_rgba[..., :3],
                     mask=inst_rgba[..., 3] > 32))
    Image.fromarray(inst_rgba).convert("RGBA").resize((x2 - x1, y2 - y1), Image.LANCZOS).save(
        out / "instance_placed.png")

    composed = host_np.copy()
    alpha_full = _alpha_paste(composed, inst_rgba, bbox)
    corrected = lab_histogram_match_local(composed, alpha_full, bbox)
    Image.fromarray(corrected).save(out / "corrected.png")

    (out / "meta.json").write_text(json.dumps({
        "host_file": ip.name, "W": host.W, "H": host.H,
        "gt_boxes": [list(map(int, b)) for b in host.gt_boxes_xyxy],
        "gt_classes": host.gt_classes,
        "target": {"cls": cls, "scale": ts, "cx": tcx, "cy": tcy},
        "scale_method": method, "bbox": [int(v) for v in bbox],
        "chi2": float(chi2), "instance": Path(inst_path).name, "accept": bool(host.accept),
    }, indent=2, ensure_ascii=False))
    print("WROTE", out)


if __name__ == "__main__":
    main()
