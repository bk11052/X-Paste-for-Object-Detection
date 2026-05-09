"""
Remap Roboflow military dataset labels to one of our YOLO schemes.

Two schemes are supported via --scheme:

  three_class  (legacy "soldier.v1i.yolov11" 12-class -> 3-class):
    Source: 0:'-', 1:'armored_car', 2:'battery', 3:'gun_barrel', 4:'gunship',
            5:'passerby', 6:'shelter_car', 7:'soldier', 8:'tank', 9:'tent',
            10:'track', 11:'uav'
    Output: 0:'tank', 1:'soldier', 2:'military_vehicle'
    Mapping: 8->0, 7->1, 1->2, 6->2, others dropped.

  four_class  (new "Custom Object Detection -Military- v1" 4-class -> identity):
    Source = Output: 0:'Soldier', 1:'civilian_vehicle', 2:'military_vehicle', 3:'persons'
    Mapping: identity (no class id changes). The script just standardizes the
    directory layout under output_root and links images.

For each source split (train/valid/test), we write labels and symlink (or copy)
matching images. Images that have no relevant labels remain (as background-only
samples) -- useful for measuring false positives.

Usage:
  # 3-class legacy
  python tools/remap_roboflow_labels.py --scheme three_class \\
      --input_root /path/to/soldier.v1i.yolov11 \\
      --output_root data/military_yolo/real

  # 4-class new (identity)
  python tools/remap_roboflow_labels.py --scheme four_class \\
      --input_root /path/to/Custom_Object_Detection_Military_v1 \\
      --output_root data/military_v1/real
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


# Map: roboflow class id -> our class id (or None to drop)
ROBOFLOW_TO_OURS_3CLS = {
    8: 0,   # tank
    7: 1,   # soldier
    1: 2,   # armored_car -> military_vehicle
    6: 2,   # shelter_car -> military_vehicle
}

# Identity mapping for the new 4-class dataset.
ROBOFLOW_TO_OURS_4CLS = {0: 0, 1: 1, 2: 2, 3: 3}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / (stem + ext)
        if p.exists():
            return p
    return None


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.absolute(), dst)


def process_split(in_root: Path, out_root: Path, split: str, copy: bool, mapping: dict, n_classes: int) -> dict:
    in_labels = in_root / split / "labels"
    in_images = in_root / split / "images"
    if not in_labels.exists() or not in_images.exists():
        return {"split": split, "status": "missing"}

    out_labels = out_root / split / "labels"
    out_images = out_root / split / "images"
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    processed = 0
    kept_with_labels = 0
    kept_empty = 0
    class_counts = {i: 0 for i in range(n_classes)}

    for txt_file in sorted(in_labels.glob("*.txt")):
        new_lines = []
        for line in txt_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            old_cls = int(parts[0])
            new_cls = mapping.get(old_cls)
            if new_cls is None:
                continue
            new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
            class_counts[new_cls] += 1

        # Always write label file (even if empty after filter) and link image
        (out_labels / txt_file.name).write_text("\n".join(new_lines))
        src_img = find_image(in_images, txt_file.stem)
        if src_img is None:
            print(f"  ! image missing for {txt_file.name}")
            continue
        link_or_copy(src_img, out_images / src_img.name, copy)
        processed += 1
        if new_lines:
            kept_with_labels += 1
        else:
            kept_empty += 1

    return {
        "split": split, "processed": processed,
        "with_labels": kept_with_labels, "empty": kept_empty,
        "class_counts": class_counts,
    }


SCHEME_NAMES = {
    "three_class": ["tank", "soldier", "military_vehicle"],
    "four_class":  ["Soldier", "civilian_vehicle", "military_vehicle", "persons"],
}
SCHEME_MAPPINGS = {
    "three_class": ROBOFLOW_TO_OURS_3CLS,
    "four_class":  ROBOFLOW_TO_OURS_4CLS,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_root", required=True, help="path containing train/, valid/, test/")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--scheme", choices=list(SCHEME_NAMES.keys()), default="three_class",
                    help="which class scheme to use (default: three_class for legacy)")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    names = SCHEME_NAMES[args.scheme]
    mapping = SCHEME_MAPPINGS[args.scheme]
    n_classes = len(names)

    for split in ["train", "valid", "test"]:
        result = process_split(in_root, out_root, split, args.copy, mapping, n_classes)
        if result.get("status") == "missing":
            print(f"[skip] {split}: missing")
            continue
        cc = result["class_counts"]
        cc_str = " ".join(f"{names[i]}={cc[i]}" for i in range(n_classes))
        print(
            f"[{split:5s}] processed={result['processed']}  "
            f"with_labels={result['with_labels']}  empty={result['empty']}  {cc_str}"
        )

    print(f"\nDone. Output: {out_root}  scheme={args.scheme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
