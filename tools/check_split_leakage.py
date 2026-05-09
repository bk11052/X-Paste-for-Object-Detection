"""
Diagnose train/valid/test split leakage in a raw Roboflow YOLO dataset.

Roboflow filenames follow the pattern:
    <original_stem>_jpg.rf.<32hex>.<ext>
The <original_stem> identifies the source frame/scene. If the same prefix
appears in multiple splits, those splits are not scene-disjoint and the
test mAP can be inflated by near-duplicates.

Usage:
    python tools/check_split_leakage.py \\
        --raw_root "data/Custom Object Detection -Military-.v1i.yolov8"

Reports pairwise prefix intersection sizes and the train∩test / |test|
ratio. >5% is treated as leakage requiring re-split.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
PREFIX_RE = re.compile(r"^(.+)_jpg\.rf\.[0-9a-f]+\.[a-zA-Z]+$")


def extract_prefix(filename: str) -> str:
    m = PREFIX_RE.match(filename)
    if m:
        return m.group(1)
    return Path(filename).stem


def collect_prefixes(images_dir: Path) -> dict[str, list[str]]:
    by_prefix: dict[str, list[str]] = {}
    if not images_dir.exists():
        return by_prefix
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        prefix = extract_prefix(p.name)
        by_prefix.setdefault(prefix, []).append(p.name)
    return by_prefix


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", required=True, help="path containing train/, valid/, test/ each with images/")
    args = ap.parse_args()

    raw = Path(args.raw_root)
    splits = ["train", "valid", "test"]
    prefixes: dict[str, set[str]] = {}
    n_images: dict[str, int] = {}
    n_unique: dict[str, int] = {}

    for s in splits:
        by_prefix = collect_prefixes(raw / s / "images")
        prefixes[s] = set(by_prefix.keys())
        n_images[s] = sum(len(v) for v in by_prefix.values())
        n_unique[s] = len(by_prefix)

    print(f"[counts]")
    for s in splits:
        print(f"  {s:5s}  images={n_images[s]:5d}  unique_prefixes={n_unique[s]:5d}")

    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    print(f"\n[pairwise prefix intersection]")
    for a, b in pairs:
        inter = prefixes[a] & prefixes[b]
        denom = max(len(prefixes[b]), 1)
        ratio = len(inter) / denom
        flag = "LEAK" if ratio > 0.05 else "ok"
        print(f"  {a:5s}∩{b:5s} = {len(inter):4d}  ({ratio:6.1%} of {b} prefixes)  [{flag}]")
        if inter:
            sample = sorted(inter)[:5]
            for prefix in sample:
                print(f"      e.g. {prefix}")

    train_test_ratio = len(prefixes["train"] & prefixes["test"]) / max(len(prefixes["test"]), 1)
    print(f"\n[verdict] train∩test / |test| = {train_test_ratio:.1%}", end="")
    if train_test_ratio > 0.05:
        print("  → LEAKAGE: re-split required")
        return 1
    else:
        print("  → splits look scene-disjoint at the prefix level")
        return 0


if __name__ == "__main__":
    sys.exit(main())
