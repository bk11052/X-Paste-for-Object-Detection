"""Depth-free scale heuristic (3-step fallback).

Decides target paste height in pixels for a class at a given (cx_norm, cy_norm) on
a host image. Avoids depth estimation, which transferred poorly from SD images
in prior experiments.

Fallback order:
  1. Same-class GT anchor: if the host already has a GT bbox of cls (or a perspective-
     equivalent class: Soldier <-> persons), use its (cy_anchor, h_anchor) and scale
     linearly with target_cy: h = h_anchor * (target_cy / cy_anchor). Clamped to
     [0.4, 2.5] * h_anchor to limit extrapolation.
  2. cy linear regression: predict h_norm = a + b * cy_norm using per-class fit
     stored in JointHist.cy_regression. Used only if R^2 >= 0.05.
  3. Target scale fallback: convert target_scale_norm (= sqrt(area_norm)) to height
     using the class's typical aspect ratio (height >= width usually for persons,
     reverse for vehicles).

Final pixel value clamped to [16, 0.85 * H_img].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Approximate aspect (width / height) per class. These are rough priors.
CLASS_ASPECT = {
    "Soldier":          0.50,
    "persons":          0.45,
    "military_vehicle": 1.80,
    "civilian_vehicle": 1.80,
}

# Perspective equivalence for cross-class anchor lookup.
EQUIV = {
    "Soldier":          ("Soldier", "persons"),
    "persons":          ("persons", "Soldier"),
    "military_vehicle": ("military_vehicle", "civilian_vehicle"),
    "civilian_vehicle": ("civilian_vehicle", "military_vehicle"),
}


@dataclass
class ScaleResult:
    height_px: int
    width_px: int
    method: str   # "gt_anchor" | "cy_regression" | "target_scale"


def _aspect(cls: str, target_aspect: float | None = None) -> float:
    if target_aspect is not None and target_aspect > 0:
        return float(target_aspect)
    return CLASS_ASPECT.get(cls, 1.0)


def _height_to_box(h_px: int, cls: str, target_aspect: float | None) -> tuple[int, int]:
    h = max(8, int(h_px))
    a = _aspect(cls, target_aspect)
    w = max(8, int(round(h * a)))
    return h, w


def choose_scale(
    cls: str,
    target_scale_norm: float,
    target_cy_norm: float,
    H_img: int,
    gt_boxes_xyxy: list[tuple[int, int, int, int]],
    gt_classes: list[str],
    cy_regression: dict[str, tuple[float, float, float]],
    target_aspect: float | None = None,
    r2_min: float = 0.05,
) -> ScaleResult:
    """Return a target height (px) and width (px) using the fallback chain."""
    method: str
    h_px: int

    # Step 1: GT anchor of same class (or perspective-equivalent class)
    accept_classes = EQUIV.get(cls, (cls,))
    anchor_h_norm = anchor_cy_norm = None
    for box, bcls in zip(gt_boxes_xyxy, gt_classes):
        if bcls not in accept_classes:
            continue
        x1, y1, x2, y2 = box
        bh = (y2 - y1) / max(1, H_img)
        bcy = ((y1 + y2) / 2) / max(1, H_img)
        if bh > 0:
            if anchor_h_norm is None or abs(bcy - target_cy_norm) < abs(anchor_cy_norm - target_cy_norm):
                anchor_h_norm = bh
                anchor_cy_norm = bcy

    if anchor_h_norm is not None and anchor_cy_norm is not None and anchor_cy_norm > 1e-3:
        ratio = max(0.4, min(2.5, target_cy_norm / anchor_cy_norm))
        h_norm = anchor_h_norm * ratio
        h_px = int(round(h_norm * H_img))
        method = "gt_anchor"
    else:
        # Step 2: cy regression
        a, b, r2 = cy_regression.get(cls, (0.0, 0.0, 0.0))
        if r2 >= r2_min:
            h_norm = max(0.0, a + b * target_cy_norm)
            if h_norm > 0:
                h_px = int(round(h_norm * H_img))
                method = "cy_regression"
            else:
                h_px = -1
                method = "target_scale"
        else:
            h_px = -1
            method = "target_scale"

        # Step 3: target_scale fallback
        if h_px <= 0 or method == "target_scale":
            asp = _aspect(cls, target_aspect)
            # area = w * h, w = h * asp -> area = h^2 * asp -> h = sqrt(area / asp)
            area_norm = max(1e-6, target_scale_norm * target_scale_norm)
            h_norm = math.sqrt(area_norm / max(1e-3, asp))
            h_px = int(round(h_norm * H_img))
            method = "target_scale"

    h_px = max(16, min(int(0.85 * H_img), int(h_px)))
    h, w = _height_to_box(h_px, cls, target_aspect)
    return ScaleResult(height_px=h, width_px=w, method=method)
