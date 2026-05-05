"""
Convert our synthetic COCO annotations.json -> YOLO format labels with train/val split.

YOLO format: one .txt per image, lines = "<class_id> <cx> <cy> <w> <h>" normalized [0,1].
Class ids are 0-indexed (COCO ids in our pipeline are 1=tank, 2=soldier, 3=military_vehicle
 -> YOLO 0=tank, 1=soldier, 2=military_vehicle).

By default, splits images randomly with seed 42:
  --val_ratio 0.1  -> 90% train / 10% val
Set --val_ratio 0 to put everything under train/.

Usage:
  python tools/coco_to_yolo.py \
      --coco output/composed_train_v2/annotations.json \
      --images_root output/composed_train_v2/images \
      --out_root data/military_yolo/synth \
      --val_ratio 0.1 --seed 42 \
      [--copy_images]      # default symlinks; --copy_images for actual copies
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.absolute(), dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images_root", required=True, help="root of input images (file_name in COCO is relative to this)")
    ap.add_argument("--out_root", required=True, help="output root; creates train/ and val/ subdirs (each with images/ + labels/)")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="fraction of images for val split (0 = all train)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy_images", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    coco = json.loads(Path(args.coco).read_text())
    img_meta = {img["id"]: img for img in coco["images"]}

    cats_sorted = sorted(coco["categories"], key=lambda c: c["id"])
    coco_to_yolo = {c["id"]: i for i, c in enumerate(cats_sorted)}
    print("Class mapping (COCO id -> YOLO id):")
    for c in cats_sorted:
        print(f"  {c['id']:>2} {c['name']:<20} -> {coco_to_yolo[c['id']]}")

    by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    # train/val split
    image_ids = sorted(img_meta.keys())
    rng = random.Random(args.seed)
    rng.shuffle(image_ids)
    n_total = len(image_ids)
    n_val = int(round(n_total * args.val_ratio))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])
    print(f"\nSplit: train={len(train_ids)}  val={len(val_ids)}  total={n_total}  (val_ratio={args.val_ratio})")

    out_root = Path(args.out_root)
    images_root_in = Path(args.images_root)

    counts = {"train": {"images": 0, "annotations": 0, "empty": 0},
              "val":   {"images": 0, "annotations": 0, "empty": 0}}

    for img_id, img in img_meta.items():
        split = "val" if img_id in val_ids else "train"
        W, H = img["width"], img["height"]
        rel = Path(img["file_name"])  # e.g. "walking_soldier/0000__0.jpg"
        anns = by_image.get(img_id, [])

        # write label file
        txt_path = out_root / split / "labels" / rel.with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            cls = coco_to_yolo[ann["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        txt_path.write_text("\n".join(lines))

        # link/copy image
        src = (images_root_in / rel).resolve()
        dst = out_root / split / "images" / rel
        link_or_copy(src, dst, args.copy_images)

        counts[split]["images"] += 1
        counts[split]["annotations"] += len(lines)
        if not lines:
            counts[split]["empty"] += 1

    print()
    for split, c in counts.items():
        print(f"[{split:5s}] images={c['images']}  annotations={c['annotations']}  empty={c['empty']}")
    print(f"\nOutput tree: {out_root}/{{train,val}}/{{images,labels}}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
