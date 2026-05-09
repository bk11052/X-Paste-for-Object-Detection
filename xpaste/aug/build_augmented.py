"""Top-level offline driver: produce an augmented YOLO training split.

For each host real image:
  1. Run HostSceneAnalyzer -> region masks + GT boxes -> accept/reject.
  2. Sample N paste configs from InverseFreqSampler (or uniform / pool_random / real_random).
  3. For each config:
       - Choose target paste height/width via scale_heuristic.choose_scale (with GT anchors).
       - Plan placement bbox using region masks + frame containment + overlap reject.
       - Pick instance crop:
           style_full     : Lab-hist top-k from SD pool
           scene_uniform  : random from SD pool (filtered by aspect/height)
           pool_random    : random from SD pool (no filter)
           real_random    : random crop from another host image of same class
       - Resize, alpha-blend onto host (custom_cp_method.blend_image).
       - Optional post Lab L-channel match (style_full only).
  4. Append accepted bboxes to a YOLO label file alongside the original GT.
  5. Save augmented image (under out_root/images) and label (out_root/labels).
  Original real images that are accepted produce 1 augmented copy (default
  --variants_per_image 1). Rejected images are still copied (no paste) so the
  output split has the same image count as the input.

CLI:
  python -m xpaste.aug.build_augmented \\
    --host_root data/military_v1/real/train \\
    --pool_dir output/pool_v1/rgba_filtered \\
    --pool_index cache/pool_index_v1.npz \\
    --hist cache/dist_v1.json \\
    --paste_mode style_full --pastes_per_image 3 --temperature 1.0 \\
    --out_root data/military_v1/aug_E/train --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import CLASSES, CLASS_TO_IDX
from .distribution import (
    InverseFreqSampler, JointHist, N_CX_BINS, N_CY_BINS, _scale_bin, load_hist,
)
from .host_scene import HostScene, HostSceneAnalyzer
from .scale_heuristic import CLASS_ASPECT, choose_scale
from .style_match import (
    PoolIndex, hist_chi2, lab_histogram_match_local, load_index,
    rgb_to_lab_bins, select_topk_match,
)


PASTE_MODES = ("style_full", "scene_uniform", "pool_random", "real_random")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class AcceptedPaste:
    cls: str
    bbox_xyxy: tuple[int, int, int, int]
    method: str
    instance_path: str | None
    sampled_bin: tuple[int, int, int, int] | None = None  # (cls_idx, scale_bin, cx_bin, cy_bin)
    target_scale: float = 0.0
    target_cx: float = 0.0
    target_cy: float = 0.0


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1); bb = (bx2 - bx1) * (by2 - by1)
    return inter / float(aa + bb - inter)


def _clip_bbox(bbox, W, H):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return (x1, y1, x2, y2)


def _bbox_inside_ratio(bbox, W, H):
    x1, y1, x2, y2 = bbox
    orig = (x2 - x1) * (y2 - y1)
    if orig <= 0:
        return 0.0
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(W, x2), min(H, y2)
    inside = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inside / orig


def _sample_anchor(mask: np.ndarray, target_cx: float, target_cy: float, rng: np.random.Generator,
                   max_attempts: int = 200) -> tuple[int, int] | None:
    """Sample a (x, y) pixel from True mask, biased toward (target_cx, target_cy).

    We sample n random valid pixels and return the one closest to (target_cx, target_cy).
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    n = min(max_attempts, xs.size)
    idx = rng.choice(xs.size, size=n, replace=False)
    cands_x = xs[idx]
    cands_y = ys[idx]
    H, W = mask.shape
    tx = target_cx * W
    ty = target_cy * H
    d2 = (cands_x - tx) ** 2 + (cands_y - ty) ** 2
    best = int(np.argmin(d2))
    return int(cands_x[best]), int(cands_y[best])


def _plan_bbox(
    scene: HostScene, cls: str, target_cx: float, target_cy: float,
    h_px: int, w_px: int, existing: list[tuple[int, int, int, int]],
    rng: np.random.Generator, max_attempts: int = 30,
    iou_thr: float = 0.10, min_inside: float = 0.92,
) -> tuple[int, int, int, int] | None:
    """Find a bbox placement near (target_cx, target_cy) inside ground/road that
    has low overlap with any existing bbox and is mostly inside frame."""
    paste_mask = scene.region_masks.get("ground", np.zeros((scene.H, scene.W), dtype=bool)) | \
        scene.region_masks.get("road", np.zeros((scene.H, scene.W), dtype=bool))
    if not paste_mask.any():
        return None
    for attempt in range(max_attempts):
        anchor = _sample_anchor(paste_mask, target_cx, target_cy, rng,
                                max_attempts=200 if attempt == 0 else 50)
        if anchor is None:
            return None
        ax, ay = anchor
        x1 = ax - w_px // 2
        x2 = x1 + w_px
        y2 = ay
        y1 = y2 - h_px
        bbox = (x1, y1, x2, y2)
        if _bbox_inside_ratio(bbox, scene.W, scene.H) < min_inside:
            continue
        clipped = _clip_bbox(bbox, scene.W, scene.H)
        if clipped is None:
            continue
        if any(_iou(clipped, e) > iou_thr for e in existing):
            continue
        return clipped
    return None


def _load_real_pool(host_root: Path) -> dict[str, list[tuple[Path, tuple[int, int, int, int]]]]:
    """Build a per-class pool of (image_path, bbox) from real GT for real_random mode."""
    pool: dict[str, list[tuple[Path, tuple[int, int, int, int]]]] = {c: [] for c in CLASSES}
    img_dir = host_root / "images"
    label_dir = host_root / "labels"
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue
        for line in label_path.read_text().splitlines():
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
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            pool[CLASSES[cls]].append((img_path, (x1, y1, x2, y2)))
    return pool


def _alpha_paste(host_rgb: np.ndarray, instance_rgba: np.ndarray, bbox: tuple[int, int, int, int]):
    """Resize instance to bbox size and alpha-blend onto host (in-place RGB array)."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    inst = Image.fromarray(instance_rgba).convert("RGBA").resize((bw, bh), Image.LANCZOS)
    inst_arr = np.asarray(inst).astype(np.float32)
    rgb = inst_arr[..., :3]
    alpha = inst_arr[..., 3:4] / 255.0
    region = host_rgb[y1:y2, x1:x2].astype(np.float32)
    blended = region * (1 - alpha) + rgb * alpha
    host_rgb[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    full_alpha = np.zeros((host_rgb.shape[0], host_rgb.shape[1]), dtype=np.float32)
    full_alpha[y1:y2, x1:x2] = alpha[..., 0]
    return full_alpha


def _crop_real_instance(rec: tuple[Path, tuple[int, int, int, int]]) -> np.ndarray | None:
    img_path, bbox = rec
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None
    x1, y1, x2, y2 = bbox
    crop = np.asarray(img.crop((x1, y1, x2, y2)))
    h, w = crop.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = crop
    rgba[..., 3] = 255  # opaque (no segmentation for real_random)
    return rgba


def _select_instance(
    mode: str, cls: str, host_crop_rgb: np.ndarray, target_aspect: float,
    target_h_px: int, pool: PoolIndex, real_pool: dict | None, rng: np.random.Generator,
):
    """Returns (rgba_array, source_path_string)."""
    if mode == "real_random":
        recs = real_pool.get(cls, []) if real_pool else []
        if not recs:
            return None, None
        rec = recs[int(rng.integers(0, len(recs)))]
        rgba = _crop_real_instance(rec)
        return rgba, str(rec[0])

    pool_recs = pool.by_class.get(cls, [])
    if not pool_recs:
        return None, None

    if mode == "pool_random":
        rec = pool_recs[int(rng.integers(0, len(pool_recs)))]
    elif mode == "scene_uniform":
        cands = [r for r in pool_recs
                 if 0.7 * target_aspect <= r.aspect <= 1.3 * target_aspect
                 and r.height_px >= 0.5 * target_h_px]
        if not cands:
            cands = pool_recs
        rec = cands[int(rng.integers(0, len(cands)))]
    elif mode == "style_full":
        rec = select_topk_match(host_crop_rgb, pool, cls, target_aspect, target_h_px, rng, k=8)
        if rec is None:
            return None, None
    else:
        raise ValueError(f"unknown paste_mode: {mode}")

    try:
        rgba = np.asarray(Image.open(rec.path).convert("RGBA"))
    except Exception:
        return None, None
    return rgba, rec.path


def _xyxy_to_yolo(bbox: tuple[int, int, int, int], W: int, H: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return cx, cy, w, h


def _truncated_normal_int(mean: float, sigma: float, lo: int, hi: int, rng: np.random.Generator) -> int:
    for _ in range(20):
        x = int(round(rng.normal(mean, sigma)))
        if lo <= x <= hi:
            return x
    return max(lo, min(hi, int(round(mean))))


def process_host(
    img_path: Path, label_path: Path, host: HostScene,
    paste_mode: str, sampler: InverseFreqSampler | None, hist: JointHist,
    pool: PoolIndex, real_pool: dict | None,
    pastes_per_image_mean: float, pastes_per_image_sigma: float,
    max_total_boxes: int, rng: np.random.Generator,
    enable_post_lab_match: bool,
) -> tuple[np.ndarray, list[AcceptedPaste]]:
    img_rgb = host.image_np.copy()
    if img_rgb.ndim == 2 or img_rgb.shape[-1] != 3:
        img_rgb = np.asarray(host.image.convert("RGB")).copy()

    accepted: list[AcceptedPaste] = []
    existing = list(host.gt_boxes_xyxy)
    if not host.accept:
        return img_rgb, accepted

    n_attempts = _truncated_normal_int(
        pastes_per_image_mean, pastes_per_image_sigma, 1, 5, rng,
    )

    for _ in range(n_attempts):
        if len(existing) >= max_total_boxes:
            break

        # Sample a paste config
        if paste_mode in ("style_full", "scene_uniform"):
            # scene_uniform: still samples a class, but uniform across cells
            if paste_mode == "scene_uniform":
                # uniform class & uniform location/scale
                cls = CLASSES[int(rng.integers(0, len(CLASSES)))]
                target_scale = float(rng.uniform(0.05, 0.3))
                target_cx = float(rng.uniform(0.1, 0.9))
                target_cy = float(rng.uniform(0.4, 0.9))
            else:
                assert sampler is not None
                cls, target_scale, target_cx, target_cy = sampler.sample(rng)
        else:  # pool_random / real_random: uniform class, random locations
            cls = CLASSES[int(rng.integers(0, len(CLASSES)))]
            target_scale = float(rng.uniform(0.05, 0.3))
            target_cx = float(rng.uniform(0.1, 0.9))
            target_cy = float(rng.uniform(0.4, 0.9))

        # Recover joint-hist bin index from continuous values for provenance.
        cls_idx = CLASS_TO_IDX[cls]
        sb = _scale_bin(target_scale, hist.scale_quartiles[cls])
        cxb = int(np.clip(target_cx * N_CX_BINS, 0, N_CX_BINS - 1))
        cyb = int(np.clip(target_cy * N_CY_BINS, 0, N_CY_BINS - 1))
        sampled_bin = (cls_idx, sb, cxb, cyb)

        target_aspect = CLASS_ASPECT.get(cls, 1.0)

        # Decide target height/width
        sr = choose_scale(
            cls=cls,
            target_scale_norm=target_scale,
            target_cy_norm=target_cy,
            H_img=host.H,
            gt_boxes_xyxy=host.gt_boxes_xyxy,
            gt_classes=host.gt_classes,
            cy_regression=hist.cy_regression,
            target_aspect=target_aspect,
        )

        # Plan placement
        if paste_mode in ("pool_random", "real_random"):
            # No region constraint -> entire image
            full_mask = np.ones((host.H, host.W), dtype=bool)
            for _try in range(30):
                anchor = _sample_anchor(full_mask, target_cx, target_cy, rng, max_attempts=200)
                if anchor is None:
                    break
                ax, ay = anchor
                x1 = ax - sr.width_px // 2
                x2 = x1 + sr.width_px
                y2 = ay
                y1 = y2 - sr.height_px
                bbox = (x1, y1, x2, y2)
                if _bbox_inside_ratio(bbox, host.W, host.H) < 0.92:
                    continue
                clipped = _clip_bbox(bbox, host.W, host.H)
                if clipped is None:
                    continue
                if any(_iou(clipped, e) > 0.10 for e in existing):
                    continue
                bbox = clipped
                break
            else:
                continue
        else:
            bbox = _plan_bbox(
                host, cls, target_cx, target_cy, sr.height_px, sr.width_px,
                existing=existing, rng=rng, max_attempts=30,
            )
            if bbox is None:
                continue

        # Get host crop for style match
        x1, y1, x2, y2 = bbox
        host_crop = img_rgb[y1:y2, x1:x2]

        # Pick instance
        rgba, src_path = _select_instance(
            paste_mode, cls, host_crop, target_aspect, sr.height_px, pool, real_pool, rng,
        )
        if rgba is None:
            continue

        # Paste
        alpha_full = _alpha_paste(img_rgb, rgba, bbox)

        # Post Lab match
        if enable_post_lab_match and paste_mode == "style_full":
            img_rgb = lab_histogram_match_local(img_rgb, alpha_full, bbox)

        existing.append(bbox)
        accepted.append(AcceptedPaste(
            cls=cls, bbox_xyxy=bbox, method=sr.method, instance_path=src_path,
            sampled_bin=sampled_bin,
            target_scale=target_scale, target_cx=target_cx, target_cy=target_cy,
        ))

    return img_rgb, accepted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host_root", required=True, help="dir containing images/ and labels/")
    ap.add_argument("--out_root", required=True, help="output dir; will create images/ and labels/")
    ap.add_argument("--paste_mode", choices=PASTE_MODES, default="style_full")
    ap.add_argument("--pool_dir", default=None, help="SD instance pool dir (for non-real_random modes)")
    ap.add_argument("--pool_index", default=None, help="cached pool index .npz")
    ap.add_argument("--hist", default=None, help="cached distribution JSON")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--pastes_per_image", type=float, default=3.0, help="mean pastes per image")
    ap.add_argument("--pastes_sigma", type=float, default=1.0)
    ap.add_argument("--max_total_boxes", type=int, default=6, help="cap GT+pasted bboxes per image")
    ap.add_argument("--variants_per_image", type=int, default=1)
    ap.add_argument("--min_paste_region_area_frac", type=float, default=0.10)
    ap.add_argument("--max_gt_area_frac", type=float, default=0.50)
    ap.add_argument("--max_gt_count", type=int, default=8)
    ap.add_argument("--no_post_lab_match", action="store_true")
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save_meta", action="store_true",
                    help="dump per-image paste provenance to <out_root>/meta.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    host_root = Path(args.host_root)
    out_root = Path(args.out_root)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    # Load distribution
    if args.hist:
        hist = load_hist(Path(args.hist))
    else:
        from .distribution import compute_joint_histogram
        hist = compute_joint_histogram(host_root / "labels")
    sampler = InverseFreqSampler(hist, temperature=args.temperature)

    # Load pool
    pool = PoolIndex()
    real_pool = None
    if args.paste_mode == "real_random":
        real_pool = _load_real_pool(host_root)
        print(f"real_pool sizes: " + ", ".join(f"{c}={len(v)}" for c, v in real_pool.items()))
    else:
        if not (args.pool_dir or args.pool_index):
            raise SystemExit("non-real_random mode requires --pool_dir or --pool_index")
        if args.pool_index and Path(args.pool_index).exists():
            pool = load_index(Path(args.pool_index))
        else:
            from .style_match import index_pool
            pool = index_pool(Path(args.pool_dir))
        print(f"pool sizes: " + ", ".join(f"{c}={len(v)}" for c, v in pool.by_class.items()))

    # Host scene analyzer (skip for non-scene_aware modes that don't need it for accept/reject)
    needs_seg = args.paste_mode in ("style_full", "scene_uniform")
    hsa = HostSceneAnalyzer() if needs_seg else None

    rng = np.random.default_rng(args.seed)

    img_dir = host_root / "images"
    label_dir = host_root / "labels"
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    n_processed = 0
    n_accepted_imgs = 0
    n_pastes_total = 0

    meta_fp = None
    if args.save_meta:
        meta_path = out_root / "meta.jsonl"
        meta_fp = meta_path.open("w")

    for img_path in img_paths:
        label_path = label_dir / (img_path.stem + ".txt")

        # Build a HostScene (with or without segformer)
        if hsa is not None:
            host = hsa.analyze(
                img_path, label_path,
                min_paste_region_area_frac=args.min_paste_region_area_frac,
                max_gt_area_frac=args.max_gt_area_frac,
                max_gt_count=args.max_gt_count,
            )
        else:
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            from .host_scene import yolo_to_xyxy
            gt_boxes, gt_classes = yolo_to_xyxy(label_path, pil.size[0], pil.size[1])
            empty_mask = np.zeros((pil.size[1], pil.size[0]), dtype=bool)
            host = HostScene(
                image=pil, image_np=np.asarray(pil),
                H=pil.size[1], W=pil.size[0],
                region_masks={"ground": empty_mask, "road": empty_mask},
                gt_boxes_xyxy=gt_boxes, gt_classes=gt_classes,
                accept=True, reject_reason=None,
            )

        for variant in range(args.variants_per_image):
            out_stem = img_path.stem if args.variants_per_image == 1 else f"{img_path.stem}__v{variant}"
            out_img_path = out_root / "images" / (out_stem + img_path.suffix)
            out_label_path = out_root / "labels" / (out_stem + ".txt")

            new_img, accepted = process_host(
                img_path, label_path, host,
                paste_mode=args.paste_mode, sampler=sampler, hist=hist,
                pool=pool, real_pool=real_pool,
                pastes_per_image_mean=args.pastes_per_image,
                pastes_per_image_sigma=args.pastes_sigma,
                max_total_boxes=args.max_total_boxes,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
                enable_post_lab_match=not args.no_post_lab_match,
            )
            n_processed += 1
            if accepted:
                n_accepted_imgs += 1
                n_pastes_total += len(accepted)
            Image.fromarray(new_img).save(out_img_path, quality=92)

            # Combine GT + new pastes into YOLO label
            lines = []
            if label_path.exists():
                lines.extend(label_path.read_text().splitlines())
            for ap_ in accepted:
                ci = CLASS_TO_IDX[ap_.cls]
                cx, cy, w, h = _xyxy_to_yolo(ap_.bbox_xyxy, host.W, host.H)
                lines.append(f"{ci} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            out_label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

            if meta_fp is not None:
                rec = {
                    "img": out_stem + img_path.suffix,
                    "src_img": img_path.name,
                    "H": host.H,
                    "W": host.W,
                    "host_accept": bool(host.accept),
                    "reject_reason": host.reject_reason,
                    "gt_count": len(host.gt_boxes_xyxy),
                    "gt_classes": [int(c) for c in host.gt_classes],
                    "paste_mode": args.paste_mode,
                    "pastes": [
                        {
                            "cls": ap_.cls,
                            "cls_idx": CLASS_TO_IDX[ap_.cls],
                            "bbox_xyxy": [int(v) for v in ap_.bbox_xyxy],
                            "scale_method": ap_.method,
                            "instance_src": ap_.instance_path,
                            "sampled_bin": list(ap_.sampled_bin) if ap_.sampled_bin else None,
                            "target_scale": ap_.target_scale,
                            "target_cx": ap_.target_cx,
                            "target_cy": ap_.target_cy,
                        } for ap_ in accepted
                    ],
                }
                meta_fp.write(json.dumps(rec) + "\n")

            if args.debug:
                print(f"{img_path.name}: accept={host.accept} reason={host.reject_reason} "
                      f"pastes={len(accepted)} -> {out_img_path}")

    if meta_fp is not None:
        meta_fp.close()
        print(f"meta jsonl: {out_root}/meta.jsonl")

    print(f"\nimages processed: {n_processed}")
    print(f"images with at least 1 paste: {n_accepted_imgs}")
    print(f"total pastes added: {n_pastes_total}")
    print(f"-> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
