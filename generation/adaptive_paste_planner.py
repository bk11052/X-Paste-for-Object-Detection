"""
Adaptive Paste Planner — the core novelty of the proposed pipeline.

Given a SceneAnalysis (depth + region masks from scene_analyzer.py) and a list of
instance specs from a scenario (category, pose, count, distance_bias), decide where
and how big each instance should be pasted so that the result is geometrically
plausible:

  - WHERE: only inside region masks compatible with the category
           (soldier/tank -> ground|road, car -> road|ground, ...)
  - HOW BIG: scale derived from per-pixel depth via a category-specific curve
             (near depth -> large bbox, far depth -> small bbox)
  - DISTANCE BIAS: scenario-level preference (near|mid|far) further restricts
                   the depth band before sampling. "far" produces small objects
                   automatically.

Output: list[PastePlan] with bbox in image coordinates and the pose slug to look
up in the instance pool.

This module is pure planning — it does NOT load or paste images. compose_scene.py
combines these plans with the instance pool and runs blending.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from scene_analyzer import CATEGORY_TO_REGIONS, SceneAnalysis


# Per-category bbox HEIGHT in pixels at near (depth=0) and far (depth=1).
# Tuned for 1024x576 SDXL backgrounds.
SCALE_CURVE = {
    "soldier": (180, 12),
    "tank":    (240, 20),  # tank "height" includes hull height; width ~2x
    "car":     (160, 16),
    "plane":   (140, 12),
    "helicopter": (140, 14),
}

# Aspect ratio (width / height) per category, used to derive bbox width from height.
ASPECT = {
    "soldier": 0.45,
    "tank":    2.10,
    "car":     1.80,
    "plane":   2.20,
    "helicopter": 1.60,
}

DISTANCE_BANDS = {
    "near": (0.00, 0.45),
    "mid":  (0.20, 0.80),
    "far":  (0.65, 1.00),
}


@dataclass
class InstanceSpec:
    category: str
    pose: str        # full pose string from scenarios.yaml
    count: int
    distance_bias: str  # "near"|"mid"|"far"


@dataclass
class PastePlan:
    category: str
    pose: str
    bbox_xyxy: tuple[int, int, int, int]   # x1, y1, x2, y2  (y2 = ground anchor)
    anchor_xy: tuple[int, int]              # foot/anchor point used during sampling
    depth: float                            # depth at anchor (0=near, 1=far)
    region: str                             # which region was matched
    pose_slug: str                          # category__slug(pose), matches pose_instances/ folder


@dataclass
class PlannerConfig:
    n_candidates: int = 400         # candidate pixels sampled per instance spec
    iou_threshold: float = 0.30     # max IoU between any two paste bboxes
    margin_px: int = 4              # min pixel margin between bboxes (additive)
    max_attempts_per_count: int = 50
    seed: int = 0


def slugify(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80]


def scale_from_depth(depth: float, category: str) -> int:
    """Logarithmic interpolation between near and far heights."""
    near_h, far_h = SCALE_CURVE.get(category, (160, 14))
    d = float(np.clip(depth, 0.0, 1.0))
    # log-curve makes far-side fall off faster (more realistic)
    h = near_h * ((far_h / near_h) ** d)
    return int(round(h))


def bbox_from_anchor(anchor_xy: tuple[int, int], category: str, height_px: int) -> tuple[int, int, int, int]:
    """Anchor is the FOOT/bottom-center. Build bbox extending upward by height_px."""
    cx, cy = anchor_xy
    w = max(4, int(round(height_px * ASPECT.get(category, 1.0))))
    h = max(4, height_px)
    x1 = cx - w // 2
    y1 = cy - h
    x2 = x1 + w
    y2 = cy
    return (x1, y1, x2, y2)


def clip_bbox(bbox: tuple[int, int, int, int], W: int, H: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (x1, y1, x2, y2)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / float(aa + bb - inter)


def expand_bbox(b: tuple[int, int, int, int], m: int) -> tuple[int, int, int, int]:
    return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)


def sample_anchor_pixels(mask: np.ndarray, n: int, rng: random.Random) -> list[tuple[int, int]]:
    """Sample n (x, y) pixels from True locations of `mask`."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    n = min(n, len(xs))
    idx = rng.sample(range(len(xs)), n)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def select_valid_mask(scene: SceneAnalysis, category: str, distance_bias: str) -> tuple[np.ndarray, str]:
    """OR all category-compatible region masks, then restrict by depth band.
    Returns (mask, primary_region_name)."""
    regions = CATEGORY_TO_REGIONS.get(category, ["ground"])
    primary = None
    mask = np.zeros((scene.H, scene.W), dtype=bool)
    for r in regions:
        rmask = scene.region_masks.get(r)
        if rmask is None or rmask.sum() == 0:
            continue
        mask |= rmask
        if primary is None:
            primary = r
    if primary is None:
        # fallback: use lower half of image as ground proxy
        mask = np.zeros((scene.H, scene.W), dtype=bool)
        mask[scene.H // 2:, :] = True
        primary = "fallback_lower"

    d_lo, d_hi = DISTANCE_BANDS.get(distance_bias, DISTANCE_BANDS["mid"])
    band = (scene.depth_norm >= d_lo) & (scene.depth_norm <= d_hi)
    mask = mask & band
    return mask, primary


def plan_paste(
    scene: SceneAnalysis,
    instance_specs: list[InstanceSpec],
    config: PlannerConfig | None = None,
) -> list[PastePlan]:
    """Decide bbox and pose slug for every instance to paste."""
    cfg = config or PlannerConfig()
    rng = random.Random(cfg.seed)

    plans: list[PastePlan] = []
    for spec in instance_specs:
        valid_mask, region = select_valid_mask(scene, spec.category, spec.distance_bias)
        if not valid_mask.any():
            continue
        candidates = sample_anchor_pixels(valid_mask, cfg.n_candidates, rng)

        placed = 0
        attempts = 0
        for (x, y) in candidates:
            if placed >= spec.count:
                break
            attempts += 1
            if attempts > cfg.max_attempts_per_count * spec.count:
                break

            d = float(scene.depth_norm[y, x])
            h_px = scale_from_depth(d, spec.category)
            bbox = bbox_from_anchor((x, y), spec.category, h_px)
            clipped = clip_bbox(bbox, scene.W, scene.H)
            if clipped is None:
                continue

            overlap_hit = False
            for prev in plans:
                if iou(expand_bbox(clipped, cfg.margin_px), prev.bbox_xyxy) > cfg.iou_threshold:
                    overlap_hit = True
                    break
            if overlap_hit:
                continue

            plans.append(PastePlan(
                category=spec.category, pose=spec.pose, bbox_xyxy=clipped,
                anchor_xy=(x, y), depth=d, region=region,
                pose_slug=f"{spec.category}__{slugify(spec.pose)}",
            ))
            placed += 1

    return plans


def specs_from_yaml_entry(scenario_entry: dict) -> list[InstanceSpec]:
    return [
        InstanceSpec(
            category=i["category"], pose=i["pose"],
            count=int(i["count"]), distance_bias=i.get("distance_bias", "mid"),
        )
        for i in scenario_entry.get("instances", [])
    ]
