"""
Build OOD-B (civilian) test split from COCO val2017.

Filters COCO val2017 to images containing person / car / bus / truck
annotations, remaps category ids to our 4-class space, and writes a
flat split:

    <output_root>/images/<filename>
    <output_root>/labels/<filename>.txt

COCO categories used:
    1 (person)        -> 3 (persons)
    3 (car)           -> 1 (civilian_vehicle)
    6 (bus)           -> 1 (civilian_vehicle)
    8 (truck)         -> 1 (civilian_vehicle)

Other categories are dropped. Images with no surviving annotations after
filtering are skipped. Output is optionally subsampled to --max_images
(default 500) using a deterministic seed.

Usage:
    python tools/build_coco_civilian_ood.py \\
        --coco_imgs data/coco/val2017 \\
        --coco_ann  data/coco/annotations/instances_val2017.json \\
        --output_root data/military_v1/ood_b \\
        --max_images 500 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

CLASS_NAMES = ["Soldier", "civilian_vehicle", "military_vehicle", "persons"]

# COCO category id -> our 4-class id
COCO_TO_OURS = {
    1: 3,  # person -> persons
    3: 1,  # car    -> civilian_vehicle
    6: 1,  # bus    -> civilian_vehicle
    8: 1,  # truck  -> civilian_vehicle
}


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.absolute(), dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_imgs", required=True, help="dir with val2017/*.jpg")
    ap.add_argument("--coco_ann", required=True, help="instances_val2017.json")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--max_images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    coco = json.loads(Path(args.coco_ann).read_text())

    # image_id -> {"file_name": str, "w": int, "h": int}
    images = {im["id"]: {"file_name": im["file_name"], "w": im["width"], "h": im["height"]}
              for im in coco["images"]}

    # Group annotations by image, only keeping mapped categories.
    ann_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in COCO_TO_OURS:
            continue
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    candidate_ids = sorted(ann_by_image.keys())
    rng.shuffle(candidate_ids)
    if args.max_images > 0:
        candidate_ids = candidate_ids[: args.max_images]

    out = Path(args.output_root)
    out_imgs = out / "images"
    out_lbls = out / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    coco_imgs_dir = Path(args.coco_imgs)
    n_kept = 0
    n_missing_image = 0
    class_counts = [0] * args.n_classes

    for img_id in candidate_ids:
        meta = images[img_id]
        src_img = coco_imgs_dir / meta["file_name"]
        if not src_img.exists():
            n_missing_image += 1
            continue
        w, h = meta["w"], meta["h"]
        if w <= 0 or h <= 0:
            continue

        lines = []
        for ann in ann_by_image[img_id]:
            tgt = COCO_TO_OURS[ann["category_id"]]
            x, y, bw, bh = ann["bbox"]  # COCO: x_min, y_min, w, h (pixels)
            if bw <= 0 or bh <= 0:
                continue
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            # Clip to [0, 1] in case of slight overruns.
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            lines.append(f"{tgt} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            class_counts[tgt] += 1

        if not lines:
            continue
        stem = Path(meta["file_name"]).stem
        (out_lbls / f"{stem}.txt").write_text("\n".join(lines))
        link_or_copy(src_img, out_imgs / src_img.name, args.copy)
        n_kept += 1

    print(f"[input]  COCO val2017: {len(images)} images, {len(coco['annotations'])} annotations")
    print(f"         {len(ann_by_image)} images contain target categories ({sorted(COCO_TO_OURS.keys())})")
    print(f"[sample] candidates after shuffle/cap: {len(candidate_ids)} (max={args.max_images}, seed={args.seed})")
    print(f"[output] {out}")
    print(f"  kept:    {n_kept} images")
    print(f"  missing: {n_missing_image} (image file not found)")
    print(f"\n[class instance counts in 4-class space]")
    for c in range(args.n_classes):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"cls_{c}"
        print(f"  {name:<18} {class_counts[c]:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
