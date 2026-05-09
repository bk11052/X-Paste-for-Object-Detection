"""
Build OOD-A (military) test split from a Roboflow YOLO export.

Pools all images from the source's train/valid/test, applies a flexible
class-id remap from a YAML file, and writes a single flat split:

    <output_root>/images/<filename>
    <output_root>/labels/<filename>.txt

Designed for the `yolo-datasets-ymdve/military-object-detection-uxkcn`
dataset (3 classes: military vehicle, soldier, tank), but works for any
Roboflow YOLO export given a mapping yaml.

Usage:
    python tools/build_ood_military.py \\
        --source_root data/military_v1/ood_a_raw \\
        --mapping_yaml configs/ood_military_mapping.yaml \\
        --output_root data/military_v1/ood_a
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.remap_roboflow_labels import find_image, link_or_copy  # noqa: E402


CLASS_NAMES = ["Soldier", "civilian_vehicle", "military_vehicle", "persons"]
SPLITS = ("train", "valid", "test")


def load_mapping(path: Path) -> dict[int, int]:
    cfg = yaml.safe_load(path.read_text())
    raw = cfg["mapping"] if "mapping" in cfg else cfg
    out: dict[int, int] = {}
    for k, v in raw.items():
        if v is None:
            continue
        out[int(k)] = int(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source_root", required=True, help="Roboflow export with train/, valid/, test/")
    ap.add_argument("--mapping_yaml", required=True, help="yaml with mapping: {src_id: our_id}")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    src = Path(args.source_root)
    out = Path(args.output_root)
    mapping = load_mapping(Path(args.mapping_yaml))
    print(f"[mapping] {mapping}")

    out_imgs = out / "images"
    out_lbls = out / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_kept = 0
    n_skipped_empty = 0
    n_missing_image = 0
    class_counts = [0] * args.n_classes

    for split in SPLITS:
        in_lbls = src / split / "labels"
        in_imgs = src / split / "images"
        if not in_lbls.exists() or not in_imgs.exists():
            print(f"[skip] {split}: missing")
            continue
        for txt in sorted(in_lbls.glob("*.txt")):
            n_in += 1
            new_lines = []
            for line in txt.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    src_cls = int(parts[0])
                except ValueError:
                    continue
                tgt = mapping.get(src_cls)
                if tgt is None:
                    continue
                new_lines.append(f"{tgt} " + " ".join(parts[1:]))
                class_counts[tgt] += 1

            if not new_lines:
                n_skipped_empty += 1
                continue
            img = find_image(in_imgs, txt.stem)
            if img is None:
                n_missing_image += 1
                continue

            (out_lbls / txt.name).write_text("\n".join(new_lines))
            link_or_copy(img, out_imgs / img.name, args.copy)
            n_kept += 1

    print(f"\n[input]  scanned {n_in} label files across {SPLITS}")
    print(f"[output] {out}")
    print(f"  kept:    {n_kept} images")
    print(f"  empty:   {n_skipped_empty} (no class survived remap)")
    print(f"  missing: {n_missing_image} (label without image)")
    print(f"\n[class instance counts in 4-class space]")
    for c in range(args.n_classes):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"cls_{c}"
        print(f"  {name:<18} {class_counts[c]:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
