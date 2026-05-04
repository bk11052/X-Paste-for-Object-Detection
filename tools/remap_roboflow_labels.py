"""
Remap Roboflow 12-class soldier dataset labels to our 3-class YOLO scheme.

Source classes (Roboflow):
  0:'-', 1:'armored_car', 2:'battery', 3:'gun_barrel', 4:'gunship',
  5:'passerby', 6:'shelter_car', 7:'soldier', 8:'tank', 9:'tent',
  10:'track', 11:'uav'

Our classes (3-class YOLO):
  0:'tank', 1:'soldier', 2:'military_vehicle'

Mapping:
  8 (tank)         -> 0 (tank)
  7 (soldier)      -> 1 (soldier)
  1 (armored_car)  -> 2 (military_vehicle)
  6 (shelter_car)  -> 2 (military_vehicle)
  others           -> dropped

For each source split (train/valid/test), we write filtered labels and symlink the
matching images. Images that have no relevant labels remain (as background-only
samples) -- useful for measuring false positives.

Usage:
  python tools/remap_roboflow_labels.py \
      --input_root /path/to/soldier.v1i.yolov11 \
      --output_root data/military_yolo/real \
      [--copy]                # default symlinks; use --copy for actual file copy
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


# Map: roboflow class id -> our class id (or None to drop)
ROBOFLOW_TO_OURS = {
    8: 0,   # tank
    7: 1,   # soldier
    1: 2,   # armored_car -> military_vehicle
    6: 2,   # shelter_car -> military_vehicle
}

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


def process_split(in_root: Path, out_root: Path, split: str, copy: bool) -> dict:
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
    class_counts = {0: 0, 1: 0, 2: 0}

    for txt_file in sorted(in_labels.glob("*.txt")):
        new_lines = []
        for line in txt_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            old_cls = int(parts[0])
            new_cls = ROBOFLOW_TO_OURS.get(old_cls)
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
        "tank": class_counts[0], "soldier": class_counts[1], "military_vehicle": class_counts[2],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_root", required=True, help="path containing train/, valid/, test/")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)

    for split in ["train", "valid", "test"]:
        result = process_split(in_root, out_root, split, args.copy)
        if result.get("status") == "missing":
            print(f"[skip] {split}: missing")
            continue
        print(
            f"[{split:5s}] processed={result['processed']}  "
            f"with_labels={result['with_labels']}  empty={result['empty']}  "
            f"tank={result['tank']} soldier={result['soldier']} mil_veh={result['military_vehicle']}"
        )

    print(f"\nDone. Output: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
