"""
Re-split a raw Roboflow YOLO dataset into scene-disjoint train/valid/test.

Pools images from all input splits, groups by prefix (origin frame), then
greedily assigns each prefix-group to the split that is currently most
under-filled relative to the target ratio (per class). All images sharing
a prefix end up in the same split, so train ∩ test = ∅ at the prefix level.

The output preserves the Roboflow YOLO directory layout (split/images,
split/labels) so it can be fed straight into tools/remap_roboflow_labels.py.

Usage:
    python tools/scene_disjoint_split.py \\
        --raw_root "data/Custom Object Detection -Military-.v1i.yolov8" \\
        --out_root data/military_v1/real_raw_new \\
        --ratios 0.70 0.20 0.10 --seed 0
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
PREFIX_RE = re.compile(r"^(.+)_jpg\.rf\.[0-9a-f]+\.[a-zA-Z]+$")
SPLITS = ("train", "valid", "test")


def extract_prefix(filename: str) -> str:
    m = PREFIX_RE.match(filename)
    if m:
        return m.group(1)
    return Path(filename).stem


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.absolute(), dst)


def collect_groups(raw_root: Path) -> dict[str, list[tuple[Path, Path | None]]]:
    """Return {prefix: [(image_path, label_path_or_None), ...]} pooling all input splits."""
    groups: dict[str, list[tuple[Path, Path | None]]] = {}
    for s in SPLITS:
        img_dir = raw_root / s / "images"
        lbl_dir = raw_root / s / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            prefix = extract_prefix(img.name)
            lbl = lbl_dir / (img.stem + ".txt")
            groups.setdefault(prefix, []).append((img, lbl if lbl.exists() else None))
    return groups


def count_classes(label_paths: list[Path | None], n_classes: int) -> list[int]:
    counts = [0] * n_classes
    for lbl in label_paths:
        if lbl is None:
            continue
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if 0 <= cid < n_classes:
                counts[cid] += 1
    return counts


def assign_groups(
    groups: dict[str, list[tuple[Path, Path | None]]],
    ratios: tuple[float, float, float],
    n_classes: int,
    seed: int,
) -> dict[str, str]:
    """Greedy stratified assignment of prefix-groups to splits.

    Score for placing a group in split s:
        sum_c group_count[c] * max(0, target[s][c] - current[s][c])
    Pick the split that gains the most by absorbing this group.
    """
    rng = random.Random(seed)

    # Precompute per-group class counts and total instances.
    group_keys = list(groups.keys())
    group_counts: dict[str, list[int]] = {}
    totals = [0] * n_classes
    n_imgs_total = 0
    for k in group_keys:
        members = groups[k]
        counts = count_classes([lbl for _, lbl in members], n_classes)
        group_counts[k] = counts
        for c in range(n_classes):
            totals[c] += counts[c]
        n_imgs_total += len(members)

    targets = {s: [t * r for t in totals] for s, r in zip(SPLITS, ratios)}
    img_targets = {s: n_imgs_total * r for s, r in zip(SPLITS, ratios)}
    current = {s: [0] * n_classes for s in SPLITS}
    img_current = {s: 0 for s in SPLITS}

    # Sort groups by total instance count desc. Tie-break by image count, then
    # by a deterministic shuffle key so groups with no labels still spread out.
    rng.shuffle(group_keys)
    group_keys.sort(key=lambda k: (sum(group_counts[k]), len(groups[k])), reverse=True)

    assignment: dict[str, str] = {}
    for k in group_keys:
        gc = group_counts[k]
        n_img = len(groups[k])

        best_score = None
        best_split = None
        for s in SPLITS:
            if sum(gc) > 0:
                score = sum(gc[c] * max(0.0, targets[s][c] - current[s][c]) for c in range(n_classes))
            else:
                # Background-only group: balance by image count alone.
                score = max(0.0, img_targets[s] - img_current[s]) * n_img
            if best_score is None or score > best_score:
                best_score = score
                best_split = s
        assert best_split is not None
        assignment[k] = best_split
        for c in range(n_classes):
            current[best_split][c] += gc[c]
        img_current[best_split] += n_img

    return assignment


def materialize(
    raw_root: Path,
    out_root: Path,
    groups: dict[str, list[tuple[Path, Path | None]]],
    assignment: dict[str, str],
    copy: bool,
) -> dict[str, dict]:
    stats: dict[str, dict] = {s: {"n_images": 0, "prefixes": set()} for s in SPLITS}
    for s in SPLITS:
        (out_root / s / "images").mkdir(parents=True, exist_ok=True)
        (out_root / s / "labels").mkdir(parents=True, exist_ok=True)

    for prefix, members in groups.items():
        s = assignment[prefix]
        stats[s]["prefixes"].add(prefix)
        for img, lbl in members:
            link_or_copy(img, out_root / s / "images" / img.name, copy)
            if lbl is not None:
                shutil.copy2(lbl, out_root / s / "labels" / lbl.name)
            else:
                # Empty label so YOLO treats it as background-only.
                (out_root / s / "labels" / (img.stem + ".txt")).write_text("")
            stats[s]["n_images"] += 1
    return stats


CLASS_NAMES = ["Soldier", "civilian_vehicle", "military_vehicle", "persons"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", required=True, help="path with train/, valid/, test/ each holding images/, labels/")
    ap.add_argument("--out_root", required=True, help="output path; will be created")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.20, 0.10],
                    metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        print(f"[error] ratios must sum to 1.0, got {sum(args.ratios)}", file=sys.stderr)
        return 2

    raw = Path(args.raw_root)
    out = Path(args.out_root)
    groups = collect_groups(raw)
    if not groups:
        print(f"[error] no images found under {raw}", file=sys.stderr)
        return 2

    n_groups = len(groups)
    n_imgs = sum(len(v) for v in groups.values())
    print(f"[input]  {n_imgs} images in {n_groups} prefix groups under {raw}")

    assignment = assign_groups(groups, tuple(args.ratios), args.n_classes, args.seed)
    stats = materialize(raw, out, groups, assignment, args.copy)

    # Recompute class instances per split for reporting.
    cls_per_split = {s: [0] * args.n_classes for s in SPLITS}
    for prefix, members in groups.items():
        s = assignment[prefix]
        for c, n in enumerate(count_classes([lbl for _, lbl in members], args.n_classes)):
            cls_per_split[s][c] += n

    print(f"\n[split sizes]")
    for s in SPLITS:
        print(f"  {s:5s}  images={stats[s]['n_images']:5d}  prefixes={len(stats[s]['prefixes']):5d}")

    print(f"\n[class balance]")
    name_w = max(len(n) for n in CLASS_NAMES[: args.n_classes])
    for c in range(args.n_classes):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"cls_{c}"
        per = "  ".join(f"{s}={cls_per_split[s][c]:4d}" for s in SPLITS)
        print(f"  {name:<{name_w}}  {per}")

    # Verify scene-disjoint at the prefix level.
    inter_tv = stats["train"]["prefixes"] & stats["valid"]["prefixes"]
    inter_tt = stats["train"]["prefixes"] & stats["test"]["prefixes"]
    inter_vt = stats["valid"]["prefixes"] & stats["test"]["prefixes"]
    print(f"\n[prefix overlap] train∩valid={len(inter_tv)}  train∩test={len(inter_tt)}  valid∩test={len(inter_vt)}", end="")
    if not (inter_tv or inter_tt or inter_vt):
        print("  ✓ scene-disjoint")
        ret = 0
    else:
        print("  ✗ overlap detected (bug)")
        ret = 1

    print(f"\nDone. Output: {out}")
    return ret


if __name__ == "__main__":
    sys.exit(main())
