# Copyright (c) Facebook, Inc. and its affiliates.

# Keep xpaste importable even when Detectron2/CenterNet compiled ops are absent.
# This is needed for the lightweight xpaste.aug utilities used in the YOLO pipeline.
try:
    from .modeling.meta_arch import custom_rcnn
    from .modeling.roi_heads import detic_roi_heads
    from .modeling.roi_heads import res5_roi_heads
    from .modeling.backbone import swintransformer
    from .modeling.backbone import timm

    from .data.datasets import lvis_v1
    from .data.datasets import syn4det
    from .ema import ModelEma
except Exception:
    pass
