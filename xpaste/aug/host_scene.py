"""Host real-image scene analyzer (SegFormer ADE20K only) + accept/reject filter.

Wraps generation/scene_analyzer.py SceneAnalyzer with use_depth=False and adds:
  - GT bbox extraction from a YOLO label file
  - host_is_acceptable() image-level rejection criteria

Reject reasons:
  1. paste-region (ground OR road) area fraction below threshold (default 0.10)
     -> indoor / sky-only / extreme close-up images
  2. existing GT bbox area fraction above threshold (default 0.50)
     -> already crowded; pasting more would only occlude
  3. existing GT bbox count above threshold (default 8)
     -> dense formation shots; bbox conflicts likely
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import CLASSES


@dataclass
class HostScene:
    image: Image.Image
    image_np: np.ndarray
    H: int
    W: int
    region_masks: dict[str, np.ndarray]
    gt_boxes_xyxy: list[tuple[int, int, int, int]]
    gt_classes: list[str]
    accept: bool
    reject_reason: str | None


def yolo_to_xyxy(label_path: Path, W: int, H: int) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    boxes, names = [], []
    if not label_path.exists():
        return boxes, names
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            continue
        if cls < 0 or cls >= len(CLASSES):
            continue
        x1 = int(round((cx - w / 2) * W))
        y1 = int(round((cy - h / 2) * H))
        x2 = int(round((cx + w / 2) * W))
        y2 = int(round((cy + h / 2) * H))
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append((x1, y1, x2, y2))
        names.append(CLASSES[cls])
    return boxes, names


def host_is_acceptable(
    region_masks: dict[str, np.ndarray],
    gt_boxes: list[tuple[int, int, int, int]],
    H: int,
    W: int,
    min_paste_region_area_frac: float = 0.10,
    max_gt_area_frac: float = 0.50,
    max_gt_count: int = 8,
) -> tuple[bool, str | None]:
    paste_mask = region_masks.get("ground", np.zeros((H, W), dtype=bool)) | region_masks.get(
        "road", np.zeros((H, W), dtype=bool)
    )
    paste_frac = float(paste_mask.sum()) / float(max(1, H * W))
    if paste_frac < min_paste_region_area_frac:
        return False, f"insufficient_paste_region (ground+road={paste_frac:.3f})"

    gt_area = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in gt_boxes)
    gt_frac = float(gt_area) / float(max(1, H * W))
    if gt_frac > max_gt_area_frac:
        return False, f"gt_area_too_dense ({gt_frac:.3f})"

    if len(gt_boxes) > max_gt_count:
        return False, f"gt_count_too_high ({len(gt_boxes)})"

    return True, None


class HostSceneAnalyzer:
    """Lightweight wrapper that calls SceneAnalyzer with use_depth=False."""

    def __init__(self, seg_model: str = "nvidia/segformer-b5-finetuned-ade-640-640", device: str | None = None):
        from generation.scene_analyzer import SceneAnalyzer
        self.analyzer = SceneAnalyzer(use_depth=False, seg_model=seg_model, device=device)

    def analyze(
        self,
        image_path: Path,
        label_path: Path,
        min_paste_region_area_frac: float = 0.10,
        max_gt_area_frac: float = 0.50,
        max_gt_count: int = 8,
    ) -> HostScene:
        scene = self.analyzer.analyze(image_path)
        gt_boxes, gt_classes = yolo_to_xyxy(label_path, scene.W, scene.H)
        accept, reason = host_is_acceptable(
            scene.region_masks, gt_boxes, scene.H, scene.W,
            min_paste_region_area_frac, max_gt_area_frac, max_gt_count,
        )
        return HostScene(
            image=scene.image,
            image_np=np.asarray(scene.image),
            H=scene.H, W=scene.W,
            region_masks=scene.region_masks,
            gt_boxes_xyxy=gt_boxes,
            gt_classes=gt_classes,
            accept=accept,
            reject_reason=reason,
        )


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None, help="optional viz output path")
    args = ap.parse_args()

    hsa = HostSceneAnalyzer()
    scene = hsa.analyze(Path(args.img), Path(args.label))
    print(f"image {scene.W}x{scene.H}  accept={scene.accept}  reason={scene.reject_reason}")
    print(f"gt_boxes={len(scene.gt_boxes_xyxy)} classes={scene.gt_classes}")
    for r, m in scene.region_masks.items():
        print(f"  region {r:10s} area={m.sum() / m.size:.1%}")
    if args.out:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(scene.image_np); axes[0].set_title("image"); axes[0].axis("off")
        axes[1].imshow(scene.image_np)
        for r, m in scene.region_masks.items():
            if m.any():
                axes[1].imshow(m, alpha=0.3, cmap={"ground": "Greens", "road": "Greys"}.get(r, "Reds"))
        axes[1].set_title("regions"); axes[1].axis("off")
        axes[2].imshow(scene.image_np)
        for (x1, y1, x2, y2), cls in zip(scene.gt_boxes_xyxy, scene.gt_classes):
            axes[2].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", lw=2))
            axes[2].text(x1, y1 - 4, cls, color="red", fontsize=8)
        axes[2].set_title(f"GT (accept={scene.accept})"); axes[2].axis("off")
        plt.tight_layout()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=120); plt.close()
        print(f"viz -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
