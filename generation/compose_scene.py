"""
Compose final training images by pasting pose instances onto SDXL backgrounds with
the Adaptive Paste Planner. Outputs composed images + COCO-format detection labels.

Inputs:
  --backgrounds_dir   output of gen_singleshot_scenes.py
                      structure: <bg_dir>/<scenario_name>/{0000.png, ..., metadata.json}
  --pose_pool_dir     directory of (segmented) instance PNGs grouped by pose slug
                      structure: <pool_dir>/<pose_slug>/{*.png}
                      PNGs may be RGBA (alpha = mask) or RGB on plain ~white bg
                      (we threshold to get a mask).
  --scenarios         configs/scenarios.yaml

Outputs (under --output_dir):
  images/<scenario_name>/<bg_index>__<paste_index>.jpg
  annotations.json   COCO format with categories tank=1, soldier=2, car=3
  visualizations/    (optional) bbox-overlaid debug images

Usage:
  python generation/compose_scene.py \
      --backgrounds_dir output/scenario_backgrounds \
      --pose_pool_dir output/pose_instances_seg \
      --scenarios configs/scenarios.yaml \
      --output_dir output/composed_train \
      --n_compositions_per_bg 4 \
      [--save_viz] [--only 1,3,9]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from adaptive_paste_planner import (
    PastePlan,
    PlannerConfig,
    plan_paste,
    specs_from_yaml_entry,
)
from scene_analyzer import SceneAnalyzer


CATEGORY_IDS = {"tank": 1, "soldier": 2, "car": 3}


def pil_to_rgba_with_mask(img: Image.Image, white_threshold: int = 240) -> Image.Image:
    """Ensure RGBA. If input is RGB, treat near-white pixels as background."""
    if img.mode == "RGBA":
        return img
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    near_white = (arr[..., 0] >= white_threshold) & (arr[..., 1] >= white_threshold) & (arr[..., 2] >= white_threshold)
    alpha = (~near_white).astype(np.uint8) * 255
    rgba = np.dstack([arr, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def crop_to_content(rgba: Image.Image) -> Image.Image:
    a = np.array(rgba)[..., 3]
    ys, xs = np.where(a > 16)
    if len(xs) == 0:
        return rgba
    return rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def list_pool(pool_dir: Path) -> dict[str, list[Path]]:
    pool: dict[str, list[Path]] = {}
    for sub in sorted(pool_dir.iterdir()):
        if not sub.is_dir():
            continue
        files = sorted(p for p in sub.iterdir() if p.suffix.lower() == ".png")
        if files:
            pool[sub.name] = files
    return pool


def paste_instances(bg: Image.Image, plans: list[PastePlan], pool: dict[str, list[Path]], rng: random.Random) -> tuple[Image.Image, list[PastePlan]]:
    """Paste each plan onto bg using alpha. Returns (composed_image, accepted_plans).
    Plans whose pose_slug is missing in the pool are skipped (with a warning)."""
    canvas = bg.convert("RGBA").copy()
    accepted: list[PastePlan] = []
    missing = set()
    for plan in plans:
        files = pool.get(plan.pose_slug)
        if not files:
            missing.add(plan.pose_slug)
            continue
        src_path = rng.choice(files)
        instance = Image.open(src_path)
        instance = pil_to_rgba_with_mask(instance)
        instance = crop_to_content(instance)

        x1, y1, x2, y2 = plan.bbox_xyxy
        target_w, target_h = x2 - x1, y2 - y1
        # preserve aspect ratio of the instance asset; fit within bbox
        iw, ih = instance.size
        if iw == 0 or ih == 0:
            continue
        scale = min(target_w / iw, target_h / ih)
        new_w, new_h = max(2, int(iw * scale)), max(2, int(ih * scale))
        instance = instance.resize((new_w, new_h), Image.LANCZOS)

        # bottom-center align inside bbox (anchor is foot)
        paste_x = x1 + (target_w - new_w) // 2
        paste_y = y2 - new_h
        canvas.alpha_composite(instance, dest=(max(0, paste_x), max(0, paste_y)))

        accepted.append(PastePlan(
            category=plan.category, pose=plan.pose,
            bbox_xyxy=(paste_x, paste_y, paste_x + new_w, paste_y + new_h),
            anchor_xy=plan.anchor_xy, depth=plan.depth, region=plan.region,
            pose_slug=plan.pose_slug,
        ))

    if missing:
        print(f"  ! missing pose slugs in pool: {sorted(missing)}")
    return canvas.convert("RGB"), accepted


def to_coco_annotations(image_id: int, ann_id_start: int, plans: list[PastePlan]) -> tuple[list[dict], int]:
    anns = []
    aid = ann_id_start
    for p in plans:
        x1, y1, x2, y2 = p.bbox_xyxy
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        anns.append({
            "id": aid, "image_id": image_id,
            "category_id": CATEGORY_IDS[p.category],
            "bbox": [int(x1), int(y1), int(w), int(h)],
            "area": float(w * h), "iscrowd": 0,
            "extra": {"pose_slug": p.pose_slug, "depth": p.depth, "region": p.region},
        })
        aid += 1
    return anns, aid


def draw_viz(image: Image.Image, plans: list[PastePlan]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for p in plans:
        x1, y1, x2, y2 = p.bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 2, y1 + 2), f"{p.category} d={p.depth:.2f}", fill="yellow")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backgrounds_dir", required=True)
    ap.add_argument("--pose_pool_dir", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_compositions_per_bg", type=int, default=4,
                    help="how many composed variants per background (each uses a fresh RNG seed)")
    ap.add_argument("--only", default="", help="comma-separated scenario ids")
    ap.add_argument("--save_viz", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.scenarios).read_text())
    by_id = {s["id"]: s for s in cfg["scenarios"]}
    only_ids = {int(x) for x in args.only.split(",") if x.strip()} if args.only else set(by_id.keys())

    pool = list_pool(Path(args.pose_pool_dir))
    print(f"Pool: {len(pool)} pose slugs, total {sum(len(v) for v in pool.values())} images")

    out_dir = Path(args.output_dir)
    img_out = out_dir / "images"
    viz_out = out_dir / "visualizations" if args.save_viz else None
    img_out.mkdir(parents=True, exist_ok=True)
    if viz_out:
        viz_out.mkdir(parents=True, exist_ok=True)

    print("Loading SceneAnalyzer (DepthAnything-V2 + SegFormer)...")
    sa = SceneAnalyzer()

    coco_images, coco_anns = [], []
    next_image_id, next_ann_id = 1, 1

    bg_root = Path(args.backgrounds_dir)
    for sid in sorted(only_ids):
        if sid not in by_id:
            continue
        scenario = by_id[sid]
        scenario_dir = bg_root / scenario["name"]
        if not scenario_dir.exists():
            print(f"[skip] scenario {sid}: no backgrounds at {scenario_dir}")
            continue
        bg_files = sorted(scenario_dir.glob("*.png"))
        if not bg_files:
            print(f"[skip] scenario {sid}: empty {scenario_dir}")
            continue

        instance_specs = specs_from_yaml_entry(scenario)
        out_scene_dir = img_out / scenario["name"]
        out_scene_dir.mkdir(parents=True, exist_ok=True)
        if viz_out:
            (viz_out / scenario["name"]).mkdir(parents=True, exist_ok=True)

        print(f"\n[{sid}] {scenario['name']}: {len(bg_files)} backgrounds x {args.n_compositions_per_bg} comps")
        for bg_idx, bg_path in enumerate(bg_files):
            scene = sa.analyze(bg_path)
            for k in range(args.n_compositions_per_bg):
                cfg_p = PlannerConfig(seed=args.seed + sid * 10000 + bg_idx * 100 + k)
                plans = plan_paste(scene, instance_specs, cfg_p)
                rng = random.Random(cfg_p.seed)
                composed, accepted = paste_instances(scene.image, plans, pool, rng)

                fname = f"{bg_idx:04d}__{k}.jpg"
                out_path = out_scene_dir / fname
                composed.save(out_path, quality=92)

                coco_images.append({
                    "id": next_image_id, "file_name": str(Path(scenario["name"]) / fname),
                    "width": scene.W, "height": scene.H,
                    "scenario_id": sid, "scenario": scenario["name"],
                })
                anns, next_ann_id = to_coco_annotations(next_image_id, next_ann_id, accepted)
                coco_anns.extend(anns)
                next_image_id += 1

                if viz_out:
                    draw_viz(composed, accepted).save(viz_out / scenario["name"] / fname)

    coco = {
        "images": coco_images,
        "annotations": coco_anns,
        "categories": [{"id": v, "name": k} for k, v in sorted(CATEGORY_IDS.items(), key=lambda x: x[1])],
    }
    (out_dir / "annotations.json").write_text(json.dumps(coco, indent=2, ensure_ascii=False))
    print(f"\nDone. images={len(coco_images)} annotations={len(coco_anns)}")
    print(f"  -> {out_dir}/annotations.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
