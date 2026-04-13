# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

X-Paste (ICML 2023) is an instance segmentation framework that extends Copy-Paste augmentation using Stable Diffusion and CLIP. It generates synthetic object instances with high-quality masks, then uses them to train detection models via Copy-Paste augmentation on LVIS/COCO datasets.

## Pipeline (5 Steps)

1. **Generate synthetic images** — `generation/text2im.py` uses Stable Diffusion to produce foreground objects from LVIS category names
2. **Segment foreground** — `segment_methods/reseg.py` extracts masks using one of 4 methods (CLIPSeg, UFO, U2Net, SelfReformer)
3. **Build instance pool** — `segment_methods/clean_pool.py` filters by CLIP score and mask area, outputs `LVIS_instance_pools.json`
4. **Train model** — `train_net.py` trains Cascade RCNN with Copy-Paste augmentation from the instance pool
5. **Inference** — `demo.py` runs trained model on images

## Commands

```bash
# Generate synthetic images (Step 1)
cd generation && python text2im.py --model diffusers --samples 100 --category_file <lvis_train.json> --output_dir <output>

# Segment (Step 2) — run per method: clipseg, UFO, U2Net, selfreformer
cd segment_methods && python reseg.py --input_dir <gen_dir> --output_dir <seg_dir> --seg_method U2Net --samples 100

# Filter and create pool (Step 3)
cd segment_methods && python clean_pool.py --input_dir <seg_dir> --image_dir <gen_dir> --output_file <pool.json> --min_clip 21 --min_area 0.05 --max_area 0.95

# Train (Step 4) — set DETECTRON2_DATASETS and edit INST_POOL_PATH in config
export DETECTRON2_DATASETS=/path/to/datasets
bash launch.sh --config-file configs/Xpaste_R50.yaml

# Override config params via command line
bash launch.sh --config-file configs/Xpaste_R50.yaml SOLVER.MAX_ITER 1000 SOLVER.IMS_PER_BATCH 8

# Inference (Step 5)
python demo.py --config-file configs/Xpaste_R50.yaml --input image.jpg --output out.jpg --opts MODEL.WEIGHTS <checkpoint.pth>

# Convert pretrained backbone weights to Detectron2 format
python tools/convert-thirdparty-pretrained-model-to-d2.py --path <model.pth>
```

## Architecture

### Config Inheritance
```
configs/Base-C2_L_R5021k_640b64_4x.yaml    # Base: CustomRCNN + CenterNet + DeticCascadeROIHeads
├── configs/Xpaste_R50.yaml                  # ResNet50 + instance pool (batch 64, 640px)
├── configs/Xpaste_swinL.yaml                # Swin-L + instance pool (batch 16, 896px)
├── configs/Xpaste_copypaste_R50.yaml        # ResNet50 + self copy-paste
└── configs/Xpaste_copypaste_swinL.yaml      # Swin-L + self copy-paste
```

Key X-Paste config params (defined in `xpaste/config.py`):
- `INPUT.INST_POOL` / `INPUT.INST_POOL_PATH` — enable and point to instance pool JSON
- `INPUT.USE_COPY_METHOD` — `'syn_copy'` (from pool) or `'self_copy'` (from dataset)
- `INPUT.CP_METHOD` — blending: `['basic']`, `['alpha']`, `['gaussian']`, `['poisson']`
- `INPUT.INST_POOL_FORMAT` — `'RGBA'` (with alpha channel for masking)
- `SOLVER.MODEL_EMA` — EMA decay rate (0.999 typical, 0 to disable)

### Model Architecture (train_net.py → xpaste/modeling/)
- **Meta-architecture**: `CustomRCNN` (`xpaste/modeling/meta_arch/custom_rcnn.py`) — extends Detectron2 GeneralizedRCNN with image label co-training
- **Proposal generator**: CenterNet2 (`third_party/CenterNet2/`)
- **ROI heads**: `DeticCascadeROIHeads` (`xpaste/modeling/roi_heads/detic_roi_heads.py`) — 3-stage cascade [0.6, 0.7, 0.8 IoU]
- **Backbones**: TIMM ResNet (`xpaste/modeling/backbone/timm.py`) or Swin Transformer (`xpaste/modeling/backbone/swintransformer.py`)
- **Zero-shot classifier**: CLIP-based (`xpaste/modeling/roi_heads/zero_shot_classifier.py`)

### Data Pipeline (xpaste/data/)
Training data flows through: `DatasetMapper` → `CopyPasteMapper` (wraps mapper, applies augmentation)

- `custom_build_copypaste_mapper.py` — loads instance pool, samples instances, applies copy-paste per batch
- `transforms/custom_copypaste.py` — core Copy-Paste algorithm (paste, blend, update masks, filter occluded)
- `transforms/custom_cp_method.py` — four blending methods: basic, alpha, gaussian, poisson
- `transforms/custom_augmentation_impl.py` — `EfficientDetResizeCrop` augmentation

### Instance Pool Format
The pool JSON maps category IDs to lists of RGBA image paths:
```json
{"0": ["*path/to/images/0/0.png", "*path/to/images/0/1.png"], "1": [...]}
```

## Key Dependencies
- **Detectron2** — core detection framework
- **CenterNet2** — proposal generator (in `third_party/`)
- **CLIP** (OpenAI) — zero-shot classification + semantic filtering
- **diffusers** — Stable Diffusion text-to-image
- **timm==0.4.9** — backbone model loading

## Dataset Layout
```
$DETECTRON2_DATASETS/
├── coco/
│   ├── train2017/
│   └── annotations/instances_train2017.json
└── lvis/
    ├── lvis_v1_train.json
    └── lvis_v1_val.json
```
LVIS images are shared with COCO (symlink `coco/train2017`).

Pre-computed metadata in `datasets/metadata/`: CLIP embeddings (`lvis_v1_clip_a+cname.npy`), category info (`lvis_v1_train_cat_info.json`).

## Current Work: Small Object 생성 & Copy-Paste

COCO 기준 small object (area < 32² = 1024px) 전용 augmentation 파이프라인 구축. 군사 카테고리(tank, soldier, fighter_craft) 타겟.

### 진행 현황
1. **[완료] Small object 생성** — `generation/text2im.py`에 `--prompt_template`, `--image_size` 추가
2. **[완료] SD 직접 생성 실험** — SD 1.5는 장면 내 tiny object 생성 불가 (512x512 학습 한계). 생성→축소 paste 방식 채택
3. **[진행] Bbox 단위 label 생성** — `segment_methods/gen_bbox_labels.py` (독립 스크립트, COCO JSON 출력)
4. **[예정] Small object copy-paste** — 생성된 객체를 1920x1080 배경에 20x20으로 paste

### 추가된 스크립트
- `generation/text2im.py` — `--prompt_template`, `--image_size`, 커스텀 카테고리 JSON 지원
- `generation/gen_small_object_in_scene.py` — SD 장면 내 small object 직접 생성 (실험용)
- `generation/military_categories.json` — 군사 커스텀 카테고리 (tank, soldier, fighter_craft)
- `segment_methods/gen_bbox_labels.py` — 생성 이미지에서 bbox label 추출 (COCO JSON)
- `segment_methods/visualize_bbox.py` — bbox 시각화

### 군사 카테고리 LVIS 매핑
- army_tank (id=1058, rare), fighter_jet (id=436), gun (id=523), helicopter (id=555), rifle (id=884)
- soldier는 LVIS에 없음 → 커스텀 카테고리로 생성 가능
