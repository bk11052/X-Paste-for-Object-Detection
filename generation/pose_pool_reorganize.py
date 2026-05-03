"""
Reorganize clean_pool.py output into pose_slug-keyed directories that
compose_scene.py expects.

Inputs:
  --pool_json     output of segment_methods/clean_pool.py
                  (JSON: {cid_str: ["*<output_dir>/images/<cid>/0.png", ...]})
  --results_json  results.json from gen_pose_instances.py (or any reseg.py results.json)
                  used to map cid (id-1) -> pose_slug ("name" field)
  --output_dir    target directory; will create <pose_slug>/<index>.png symlinks
                  by default, or copies if --copy is set

Result:
  <output_dir>/<pose_slug>/0000.png  (RGBA, cropped to bbox by clean_pool.py)
  ...

Usage:
  python generation/pose_pool_reorganize.py \
      --pool_json output/pose_pool/pool.json \
      --results_json output/pose_instances/results.json \
      --output_dir output/pose_pool_by_slug
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def resolve_path(stored: str, pool_root: Path) -> Path:
    # clean_pool.py stores entries like "*<full output dir>/images/<cid>/<k>.png"
    # The leading "*" is a marker; strip it. The rest is an absolute or relative path.
    p = stored[1:] if stored.startswith("*") else stored
    path = Path(p)
    if not path.is_absolute():
        path = (pool_root / path).resolve()
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool_json", required=True, help="output JSON from segment_methods/clean_pool.py")
    ap.add_argument("--results_json", required=True, help="results.json with name + id (from gen_pose_instances.py)")
    ap.add_argument("--output_dir", required=True, help="target directory keyed by pose slug")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    args = ap.parse_args()

    pool_path = Path(args.pool_json).resolve()
    pool_root = pool_path.parent
    pool = json.loads(pool_path.read_text())

    items = json.loads(Path(args.results_json).read_text())
    cid_to_slug = {str(int(it["id"]) - 1): it["name"] for it in items}

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    missing_slugs = []
    for cid_str, paths in pool.items():
        slug = cid_to_slug.get(str(cid_str))
        if slug is None:
            missing_slugs.append(cid_str)
            continue
        target_dir = out_root / slug
        target_dir.mkdir(parents=True, exist_ok=True)
        for k, p in enumerate(paths):
            src = resolve_path(p, pool_root)
            if not src.exists():
                print(f"  ! missing source: {src}")
                continue
            dst = target_dir / f"{k:04d}.png"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.copy:
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
            total_files += 1
        print(f"  [{slug}] {len(paths)} files")

    if missing_slugs:
        print(f"  ! cids without slug mapping: {missing_slugs}")
    print(f"\nDone. {total_files} files in {out_root}")
    print(f"\nNext: python generation/compose_scene.py --pose_pool_dir {out_root} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
