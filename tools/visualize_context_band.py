"""
Context-aware paste PoC — placement band 시각화

Depth-Anything V2 + SegFormer(ADE20K)로 배경 이미지에서
카테고리별 paste 가능 영역(placement band)과 샘플 위치를 그린다.

설치:
    pip install transformers accelerate Pillow matplotlib

사용법:
    python tools/visualize_context_band.py \
        --input path/to/bg.jpg \
        --output_dir output/context_band \
        --categories tank soldier fighter_jet car \
        --samples_per_cat 5

산출물 (output_dir):
    <stem>_depth.png           — depth map
    <stem>_seg.png             — semantic segmentation (ADE20K)
    <stem>_band_<category>.png — 카테고리별 binary band
    <stem>_overlay.png         — 4 카테고리 합본 + 샘플 위치 점
"""
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

# ADE20K class id (SegFormer ade-640 출력 기준)
ADE20K = {
    "sky": 2,
    "road": 6,
    "grass": 9,
    "earth": 13,
    "sand": 46,
    "field": 29,
    "path": 52,
    "dirt_track": 91,
    "runway": 96,
}


@dataclass
class CategoryPrior:
    """카테고리별 placement prior (Plan에서 정의)."""
    name: str
    seg_classes: List[str]                # ADE20K class names
    depth_range: Tuple[float, float]      # 정규화된 depth (0=원거리, 1=근거리)
    scale_log_mu: float                   # log-normal scale prior μ
    scale_log_sigma: float = 0.3
    color: Tuple[int, int, int] = (255, 0, 0)


CATEGORY_PRIORS = {
    "tank": CategoryPrior(
        name="tank",
        seg_classes=["road", "earth", "sand", "field", "path", "dirt_track"],
        depth_range=(0.3, 0.9),
        scale_log_mu=-1.5,
        color=(220, 80, 60),
    ),
    "car": CategoryPrior(
        name="car",
        seg_classes=["road", "path", "runway"],
        depth_range=(0.3, 0.95),
        scale_log_mu=-1.7,
        color=(60, 140, 220),
    ),
    "soldier": CategoryPrior(
        name="soldier",
        seg_classes=["earth", "grass", "field", "sand", "path"],
        depth_range=(0.4, 0.95),
        scale_log_mu=-2.0,
        color=(60, 200, 80),
    ),
    "fighter_jet": CategoryPrior(
        name="fighter_jet",
        seg_classes=["sky"],
        depth_range=(0.0, 0.4),
        scale_log_mu=-2.5,
        color=(220, 200, 60),
    ),
}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(device: torch.device):
    from transformers import (
        AutoImageProcessor,
        AutoModelForDepthEstimation,
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )

    depth_id = "depth-anything/Depth-Anything-V2-Small-hf"
    seg_id = "nvidia/segformer-b5-finetuned-ade-640-640"

    print(f"[load] depth: {depth_id}")
    depth_proc = AutoImageProcessor.from_pretrained(depth_id)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_id).to(device).eval()

    print(f"[load] seg: {seg_id}")
    seg_proc = SegformerImageProcessor.from_pretrained(seg_id)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_id).to(device).eval()

    return depth_proc, depth_model, seg_proc, seg_model


@torch.no_grad()
def infer_depth(image: Image.Image, proc, model, device) -> np.ndarray:
    inputs = proc(images=image, return_tensors="pt").to(device)
    out = model(**inputs)
    depth = out.predicted_depth  # (1, H', W')
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    # 정규화: 0(원거리) ~ 1(근거리)
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth)
    return (depth - d_min) / (d_max - d_min)


@torch.no_grad()
def infer_seg(image: Image.Image, proc, model, device) -> np.ndarray:
    inputs = proc(images=image, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = out.logits  # (1, C, H', W')
    logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    return logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)


def compute_placement_band(
    depth: np.ndarray, seg: np.ndarray, prior: CategoryPrior
) -> np.ndarray:
    """카테고리 prior에 맞는 binary mask (uint8: 0 or 255)."""
    seg_ids = [ADE20K[c] for c in prior.seg_classes if c in ADE20K]
    sem_mask = np.isin(seg, seg_ids)
    d_lo, d_hi = prior.depth_range
    depth_mask = (depth >= d_lo) & (depth <= d_hi)
    return (sem_mask & depth_mask).astype(np.uint8) * 255


def sample_positions(
    band: np.ndarray, prior: CategoryPrior, n: int, rng: np.random.Generator
) -> List[Tuple[int, int, float]]:
    """band 내에서 (x, y, scale) n개 샘플."""
    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        return []
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    out = []
    for i in idx:
        s = float(np.exp(rng.normal(prior.scale_log_mu, prior.scale_log_sigma)))
        out.append((int(xs[i]), int(ys[i]), s))
    return out


def colorize_depth(depth: np.ndarray) -> Image.Image:
    import matplotlib.cm as cm
    rgba = (cm.get_cmap("magma")(depth) * 255).astype(np.uint8)
    return Image.fromarray(rgba[..., :3])


def colorize_seg(seg: np.ndarray, num_classes: int = 150) -> Image.Image:
    rng = np.random.default_rng(seed=42)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return Image.fromarray(palette[seg])


def overlay_band(bg: Image.Image, band: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5) -> Image.Image:
    bg_rgba = bg.convert("RGBA")
    overlay = np.zeros((*band.shape, 4), dtype=np.uint8)
    overlay[band > 0] = (*color, int(255 * alpha))
    return Image.alpha_composite(bg_rgba, Image.fromarray(overlay))


def draw_samples(
    img: Image.Image, samples: List[Tuple[int, int, float]], prior: CategoryPrior
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    H = img.height
    for x, y, s in samples:
        r = max(4, int(s * H * 0.5))
        draw.ellipse([x - r, y - r, x + r, y + r], outline=prior.color, width=3)
        draw.text((x + r + 2, y - 8), f"{prior.name}", fill=prior.color)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="배경 이미지 경로")
    parser.add_argument("--output_dir", default="output/context_band")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["tank", "soldier", "fighter_jet", "car"],
        choices=list(CATEGORY_PRIORS.keys()),
    )
    parser.add_argument("--samples_per_cat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    rng = np.random.default_rng(args.seed)
    device = pick_device()
    print(f"[device] {device}")

    image = Image.open(args.input).convert("RGB")
    depth_proc, depth_model, seg_proc, seg_model = load_models(device)

    print("[infer] depth")
    depth = infer_depth(image, depth_proc, depth_model, device)
    print("[infer] seg")
    seg = infer_seg(image, seg_proc, seg_model, device)

    colorize_depth(depth).save(os.path.join(args.output_dir, f"{stem}_depth.png"))
    colorize_seg(seg).save(os.path.join(args.output_dir, f"{stem}_seg.png"))

    overlay = image.convert("RGBA")
    for cat in args.categories:
        prior = CATEGORY_PRIORS[cat]
        band = compute_placement_band(depth, seg, prior)
        Image.fromarray(band).save(
            os.path.join(args.output_dir, f"{stem}_band_{cat}.png")
        )
        coverage = (band > 0).mean()
        samples = sample_positions(band, prior, args.samples_per_cat, rng)
        print(f"[{cat}] band coverage={coverage:.3f}  samples={len(samples)}")
        overlay = overlay_band(overlay, band, prior.color, alpha=0.35)
        overlay = draw_samples(overlay, samples, prior)

    overlay.convert("RGB").save(os.path.join(args.output_dir, f"{stem}_overlay.png"))
    print(f"[done] -> {args.output_dir}")


if __name__ == "__main__":
    main()
