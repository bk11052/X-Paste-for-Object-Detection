"""
Convert our synthetic COCO annotations.json -> YOLO format labels.

YOLO format: one .txt per image, lines = "<class_id> <cx> <cy> <w> <h>" normalized [0,1].
Class ids are 0-indexed (COCO ids in our pipeline are 1=tank, 2=soldier, 3=military_vehicle
 -> YOLO 0=tank, 1=soldier, 2=military_vehicle).

Usage:
  python tools/coco_to_yolo.py \
      --coco output/composed_train_v2/annotations.json \
      --images_root output/composed_train_v2/images \
      --out_root data/military_yolo/synth \
      [--symlink_images]      # also create images/ tree as symlinks for ultralytics
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images_root", required=True, help="root of input images (file_name in COCO is relative to this)")
    ap.add_argument("--out_root", required=True, help="output root; creates labels/ and optionally images/")
    ap.add_argument("--symlink_images", action="store_true",
                    help="create symlinks of images under <out_root>/images mirroring labels/")
    args = ap.parse_args()

    coco = json.loads(Path(args.coco).read_text())
    img_meta = {img["id"]: img for img in coco["images"]}

    # COCO id (1-indexed) -> YOLO id (0-indexed)
    cats_sorted = sorted(coco["categories"], key=lambda c: c["id"])
    coco_to_yolo = {c["id"]: i for i, c in enumerate(cats_sorted)}
    print("Class mapping (COCO id -> YOLO id):")
    for c in cats_sorted:
        print(f"  {c['id']:>2} {c['name']:<20} -> {coco_to_yolo[c['id']]}")

    by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    out_root = Path(args.out_root)
    labels_root = out_root / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    images_root_in = Path(args.images_root)
    images_root_out = out_root / "images" if args.symlink_images else None
    if images_root_out:
        images_root_out.mkdir(parents=True, exist_ok=True)

    written = 0
    empty = 0
    for img_id, img in img_meta.items():
        W, H = img["width"], img["height"]
        rel = Path(img["file_name"])  # e.g. "walking_soldier/0000__0.jpg"
        anns = by_image.get(img_id, [])

        txt_path = labels_root / rel.with_suffix(".txt")
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
        if not lines:
            empty += 1
        written += 1

        if images_root_out:
            src = (images_root_in / rel).resolve()
            dst = images_root_out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)

    print(f"\nWrote {written} label files ({empty} empty), labels root: {labels_root}")
    if images_root_out:
        print(f"Image symlinks under: {images_root_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
