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

## 논문 계획 — 한국군사과학기술학회

### 논문 개요 (2026-05-03 방향 전환)

**Scenario-aware Copy-Paste Augmentation** — LLM이 생성한 시나리오 맥락 배경(SDXL single-shot) + 시나리오별 자세 인스턴스 풀(SD 1.5) + scene-aware paste(depth/segmentation 기반 위치/크기 자동 결정).

기존 X-Paste(랜덤 paste, 일반 배경)와 달리:
- 배경은 시나리오 맥락 반영 (전장 거리, 사막의 대치, 야간 작전 등)
- 인스턴스는 시나리오 자세 (걷는, 사격, 마주보는 등)
- Paste는 scene-aware (DepthAnything-V2 + SegFormer ADE20K)

리뷰어 대응 narrative:
- "왜 single-shot 안 함?" → SDXL single-shot은 소형 객체와 자세 제어가 약함을 정량 입증, 우리가 보완.
- "왜 X-Paste랑 다름?" → ① 시나리오 배경, ② 시나리오 자세 인스턴스, ③ scene-aware paste.

### Contributions
1. **시나리오 맥락 배경 풀** — GPT가 시나리오 프롬프트 생성 → SDXL이 객체 없는 맥락 배경을 single-shot 생성
2. **시나리오 자세 인스턴스 풀** — 기존 SD 1.5 인스턴스 풀 확장, 자세별(walking/kneeling/facing 등) prompt template 사용
3. **Scene-aware paste** — DepthAnything-V2 depth로 거리, SegFormer ADE20K로 가능 영역(ground/road/sky) 식별, depth 기반 자동 scale 산출. 소형 객체는 depth가 큰(먼) 영역에 자동 paste되어 자연 축소
4. **군사 객체 탐지 벤치마크** — DOTA + Open Images + Roboflow 통합 표준 testset

### 시나리오 카테고리 (10개 엣지 케이스)

| # | 시나리오 | 인스턴스 자세 | 도전 |
|---|---------|--------------|------|
| 1 | 걸어오는 군인 | walking_soldier × 1 | 자세 |
| 2 | 분대 군집 | mixed_soldier × 5 | 다중 |
| 3 | 마주보는 탱크 | tank_left + tank_right | 공간 관계 |
| 4 | 탱크 종대 | tank_side × 4 | 선형 배치 |
| 5 | 호송 차량 행렬 | car_side × 6 | 다수 차량 |
| 6 | 야간 정찰 | walking_soldier × 2 | 저조도 |
| 7 | 연막 속 진격 | running_soldier × 3 | 가림 |
| 8 | 위장 군인 | prone_soldier × 1 | 배경 융합 |
| 9 | 멀리 보이는 순찰대 | walking_soldier × 4 (소형) | 소형 객체 |
| 10 | 지평선의 탱크 | tank × 2 (소형) | 소형 객체 |

### 실험 설계 (6개 × 3 모델 = 18 runs)

**카테고리**: tank, soldier, car
**모델**: YOLOv8, YOLOv11, RT-DETR
**평가**: mAP + AP_small/medium/large + 카테고리별 AP

| Exp | Train | 평가 |
|-----|-------|------|
| Exp-1 | 실제 데이터 only | 실제 testset (baseline) |
| Exp-2 | X-Paste random paste (기존) | 실제 testset |
| Exp-3 | SDXL single-shot only (객체까지 SDXL) | 실제 testset |
| Exp-4 | SDXL 시나리오 배경 + random paste | 실제 testset |
| Exp-5 | **SDXL 시나리오 배경 + scene-aware paste (제안)** | 실제 testset |
| Exp-6 | Exp-5 + 실제 데이터 혼합 | 실제 testset |

핵심 비교: Exp-2 vs Exp-5 (scene-aware 효과), Exp-3 vs Exp-5 (single-shot vs 우리), Exp-4 vs Exp-5 (paste 방식 ablation), Exp-6 vs Exp-1 (보강 효과).

### 새 파이프라인 (Phase 1 — 신규 스크립트)

| 파일 | 역할 |
|------|------|
| `configs/scenarios.yaml` | 10개 시나리오 정의 (배경 프롬프트 슬롯 + 인스턴스 자세 spec) |
| `generation/gen_scenario_prompts.py` | GPT-4 API로 시나리오 배경 프롬프트 확장 + 캐시 |
| `generation/gen_singleshot_scenes.py` | SDXL `StableDiffusionXLPipeline`로 배경 생성 |
| `generation/gen_pose_instances.py` | SD 1.5 + 자세별 prompt_template으로 인스턴스 생성 (`text2im.py` 확장) |
| `generation/scene_analyzer.py` | DepthAnything-V2 + SegFormer ADE20K wrapper |
| `generation/adaptive_paste_planner.py` | depth/seg 기반 paste 위치/크기 결정 알고리즘 |
| `generation/compose_scene.py` | 통합 파이프라인 + COCO JSON 출력 |

### 재사용 자산 (기존)
- `generation/text2im.py` — `--prompt_template`, `--image_size` 지원 (자세별 인스턴스 생성에 활용)
- `generation/military_categories.json` — tank, soldier, car
- `segment_methods/reseg.py` + `clean_pool.py` — 인스턴스 세그멘테이션/필터링 (자세 인스턴스에 그대로 적용)
- `xpaste/data/transforms/custom_cp_method.py` — alpha/poisson blending
- `segment_methods/gen_bbox_labels.py` — bbox 라벨 추출 참고

### Plan 파일
세부 단계 및 알고리즘은 `/Users/kyu216/.claude/plans/rosy-splashing-whistle.md` 참조.

### 군사 카테고리 LVIS 매핑 (참고)
- army_tank (id=1058, rare), fighter_jet (id=436), gun (id=523), helicopter (id=555), rifle (id=884)
- soldier는 LVIS에 없음 → 커스텀 카테고리
