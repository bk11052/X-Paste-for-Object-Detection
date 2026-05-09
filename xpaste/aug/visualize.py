"""Bbox sanity visualizer for augmented YOLO outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import CLASSES

COLORS = {
    "Soldier": (255, 64, 64),
    "civilian_vehicle": (64, 200, 64),
    "military_vehicle": (64, 64, 255),
    "persons": (255, 200, 0),
}


def parse_yolo(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            out.append((int(parts[0]), *(float(x) for x in parts[1:5])))
        except ValueError:
            continue
    return out


def draw(img_path: Path, label_path: Path, out_path: Path) -> None:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    drw = ImageDraw.Draw(img)
    for cls, cx, cy, w, h in parse_yolo(label_path):
        if cls < 0 or cls >= len(CLASSES):
            continue
        cname = CLASSES[cls]
        x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
        col = COLORS.get(cname, (255, 0, 255))
        drw.rectangle((x1, y1, x2, y2), outline=col, width=3)
        drw.text((x1 + 2, max(0, y1 - 12)), cname, fill=col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    draw(Path(args.img), Path(args.label), Path(args.out))
    print(f"viz -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
