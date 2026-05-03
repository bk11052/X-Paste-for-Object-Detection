"""
Scene analysis: per-image depth (DepthAnything-V2) + semantic segmentation (SegFormer ADE20K).

Used by adaptive_paste_planner.py to decide:
  - WHERE objects can plausibly be pasted (segmentation: ground/road/sky regions)
  - HOW BIG each pasted object should be (depth: farther pixel -> smaller scale)

Usage as a module:
    from generation.scene_analyzer import SceneAnalyzer
    sa = SceneAnalyzer()
    scene = sa.analyze("path/to/bg.png")
    # scene.depth_norm: HxW float32 in [0,1], 0=near, 1=far
    # scene.seg_class:  HxW int32, ADE20K class id
    # scene.region_masks: dict like {"ground": HxW bool, "road": HxW bool, "sky": HxW bool}

Usage as CLI (analyze a single image and dump visualizations):
    python generation/scene_analyzer.py --image bg.png --out_dir output/scene_dbg
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ADE20K class names (id -> name) for the labels we care about.
# We map a small set of ADE classes into "paste target" regions.
ADE20K_PASTE_REGIONS = {
    "ground": [
        "earth, ground",
        "grass",
        "field",
        "sand",
        "dirt track",
        "path",
        "land, ground, soil",
        "floor",
    ],
    "road": [
        "road, route",
        "runway",
        "sidewalk, pavement",
    ],
    "sky": [
        "sky",
    ],
    "water": [
        "water",
        "sea",
        "river",
    ],
    "building": [
        "building, edifice",
        "house",
        "wall",
    ],
}

# Category -> preferred paste regions (in priority order). Used by paste planner.
CATEGORY_TO_REGIONS = {
    "soldier": ["ground", "road"],
    "tank":    ["ground", "road"],
    "car":     ["road", "ground"],
    "plane":   ["sky"],
    "helicopter": ["sky"],
}


@dataclass
class SceneAnalysis:
    image: Image.Image
    depth_norm: np.ndarray            # HxW float32 in [0,1], 0=near, 1=far
    seg_class: np.ndarray             # HxW int, ADE20K class id
    seg_id_to_name: dict              # dict id -> ADE20K class name
    region_masks: dict                # {"ground": HxW bool, ...}
    H: int
    W: int


class SceneAnalyzer:
    def __init__(
        self,
        depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
        seg_model: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        device: str | None = None,
    ):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation, SegformerForSemanticSegmentation

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.depth_proc = AutoImageProcessor.from_pretrained(depth_model)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model).to(self.device).eval()
        self.seg_proc = AutoImageProcessor.from_pretrained(seg_model)
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_model).to(self.device).eval()

        self.id2label = {int(k): v for k, v in self.seg_model.config.id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # build name lookup for region mapping (case-insensitive prefix match)
        self.region_to_ids: dict[str, list[int]] = {}
        for region, names in ADE20K_PASTE_REGIONS.items():
            ids = []
            for nm in names:
                key = nm.lower()
                for label_id, label_name in self.id2label.items():
                    if label_name.lower() == key:
                        ids.append(label_id)
                        break
                else:
                    # fallback: substring match on first word
                    short = key.split(",")[0].strip()
                    for label_id, label_name in self.id2label.items():
                        if short and short in label_name.lower():
                            ids.append(label_id)
                            break
            self.region_to_ids[region] = sorted(set(ids))

    @torch.inference_mode()
    def _predict_depth(self, img: Image.Image) -> np.ndarray:
        inputs = self.depth_proc(images=img, return_tensors="pt").to(self.device)
        out = self.depth_model(**inputs)
        depth = out.predicted_depth  # 1xHxW (relative inverse depth in DA-V2)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=img.size[::-1], mode="bicubic", align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)
        # DepthAnything-V2 outputs INVERSE relative depth (high = near). Convert to far-large.
        depth = depth.max() - depth
        # normalize to [0, 1]
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth)
        return (depth - d_min) / (d_max - d_min)

    @torch.inference_mode()
    def _predict_seg(self, img: Image.Image) -> np.ndarray:
        inputs = self.seg_proc(images=img, return_tensors="pt").to(self.device)
        out = self.seg_model(**inputs)
        logits = torch.nn.functional.interpolate(
            out.logits, size=img.size[::-1], mode="bilinear", align_corners=False,
        )
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

    def analyze(self, image: str | Path | Image.Image) -> SceneAnalysis:
        img = Image.open(image).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
        depth = self._predict_depth(img)
        seg = self._predict_seg(img)

        region_masks = {}
        for region, ids in self.region_to_ids.items():
            mask = np.zeros_like(seg, dtype=bool)
            for cid in ids:
                mask |= (seg == cid)
            region_masks[region] = mask

        return SceneAnalysis(
            image=img, depth_norm=depth, seg_class=seg,
            seg_id_to_name=self.id2label, region_masks=region_masks,
            H=img.size[1], W=img.size[0],
        )


def visualize(scene: SceneAnalysis, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    scene.image.save(out_dir / "00_image.png")

    plt.imsave(out_dir / "01_depth.png", scene.depth_norm, cmap="magma")

    fig, ax = plt.subplots(1, len(scene.region_masks), figsize=(4 * len(scene.region_masks), 4))
    if len(scene.region_masks) == 1:
        ax = [ax]
    for a, (name, mask) in zip(ax, scene.region_masks.items()):
        a.imshow(scene.image)
        a.imshow(mask, alpha=0.5)
        a.set_title(f"{name} ({mask.sum() / mask.size:.1%})")
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "02_regions.png", dpi=120)
    plt.close()

    seg_vis = (scene.seg_class.astype(np.float32) / max(1, scene.seg_class.max()))
    plt.imsave(out_dir / "03_seg.png", seg_vis, cmap="tab20")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--depth_model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--seg_model", default="nvidia/segformer-b5-finetuned-ade-640-640")
    args = ap.parse_args()

    sa = SceneAnalyzer(args.depth_model, args.seg_model)
    scene = sa.analyze(args.image)
    print(f"image: {scene.W}x{scene.H}")
    for r, m in scene.region_masks.items():
        print(f"  region {r:10s}  area={m.sum() / m.size:.1%}  ade_ids={sa.region_to_ids[r]}")
    visualize(scene, Path(args.out_dir))
    print(f"viz -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
