# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo started from **X-Paste (ICML 2023)** and has been repurposed for our paper **"Distribution-aware Generative Copy-Paste for Military Object Detection"** (한국군사과학기술학회).

**Direction (2026-05-09 onward)**: Augmentation-focused. We do **not** generate backgrounds. Instead, we paste SD 1.5–generated instances onto existing real training images. Novelty is **distribution-aware paste sampling** — bias paste configs toward underrepresented bins of the host dataset's empirical (class × scale × location) distribution, with style-matched instance selection from a diverse pool.

Detection target: 4 classes — `Soldier`, `civilian_vehicle`, `military_vehicle`, `persons`. Trained with YOLO11 (n/s/m).

(Prior direction — SDXL scenario backgrounds + scene-aware paste — failed empirically: synth-only mAP50 < 0.01, mixed only +1.6~+4.8pt. Files preserved for reference, not in active path; see "Deprecated" below.)

## Pipeline

```
[Real train labels]  ──► Distribution Analyzer (joint hist 4×4×4×4 = 256 cells)
                                    │
                                    ▼
                         Inverse-Freq Sampler (T=1.0)
                                    │
[Real train img]  ──► HostSceneAnalyzer (SegFormer ADE20K only)
                      └─► host_is_acceptable(min_paste=0.10, max_gt_area=0.50, max_gt_count=8)
                                    │
                                    ▼
                       Placement Planner (depth-free)
                       ├── region match (ground/road) via SegFormer
                       ├── scale heuristic (same-class anchor → cy regression → fallback)
                       └── frame containment + overlap reject
                                    │
[SD 1.5 instance pool, RGBA]  ──► Style-matched Selection (Lab hist χ², top-k=8)
                                    │
                                    ▼
                       alpha-blend + post Lab L-channel match
                                    │
                                    ▼
                       Augmented YOLO dataset → YOLO11 n/s/m × 3 seeds
```

## Contributions

1. **Distribution-aware paste sampling** — inverse-frequency biased sampling from joint (class, scale, location) histogram. Oversamples tail bins for true augmentation value, not just "more data".
2. **Style-matched instance selection** — Lab histogram chi-square between candidate instance and host crop; top-k closest sampled. Post-paste L-channel matching with capped shift (≤20).
3. **Depth-free scale heuristic** — three-step fallback (same-class GT anchor → cy linear regression → distribution sample). Avoids unreliable depth estimation that hurt prior runs.
4. **Class-noise mitigation** — for the Soldier-vs-persons label leakage in the dataset, narrow prompts + banned-word lists + post-generation CLIP margin filter (≥0.10).

## Dataset

Base: `Custom Object Detection -Military-.v1i.yolov8` (Roboflow, 1934 imgs, 4-class, native variable resolution 523×640 to 6720×4480, no preprocessing).

```
data/
└── military_v1/
    ├── real/{train,valid,test}/{images,labels}    # 4-class via remap_roboflow_labels.py --scheme four_class
    └── aug_{B,C,D,E}/train/{images,labels}        # 실험별 증강 train split
```

Train class instances: Soldier 577 / civilian_vehicle 424 / military_vehicle 363 / persons 856.

YOLO config (`configs/military_4cls.yaml`):
```yaml
path: /workspace/XPaste/data/military_v1
train: real/train/images        # swapped per experiment by run_yolo_matrix.sh
val:   real/valid/images
test:  real/test/images
names: { 0: Soldier, 1: civilian_vehicle, 2: military_vehicle, 3: persons }
```

## Pipeline scripts

| File | Role | Status |
|------|------|--------|
| `configs/instance_poses_4cls.yaml` | 4-class 자세/각도 spec | NEW |
| `configs/military_4cls.yaml` | YOLO data config | NEW |
| `generation/gen_pose_instances.py` | SD 1.5 자세별 인스턴스 생성 | reused |
| `generation/segment_pose_hf.py` | CLIPSeg + CLIP 필터 → RGBA | reused |
| `generation/scene_analyzer.py` | SegFormer ADE20K (use_depth=False 옵션 추가) | modified |
| `generation/adaptive_paste_planner.py` | depth=None 경로 추가 | modified |
| `tools/remap_roboflow_labels.py` | --scheme {three_class, four_class} | modified |
| `tools/filter_pool_by_clip_margin.py` | R1 mitigation 풀 필터 | NEW |
| `tools/run_yolo_matrix.sh` | 실험 행렬 드라이버 | NEW |
| `xpaste/aug/__init__.py` | aug 패키지 (CLASSES) | NEW |
| `xpaste/aug/distribution.py` | joint hist + inverse-freq sampler | NEW |
| `xpaste/aug/host_scene.py` | host SegFormer + accept/reject | NEW |
| `xpaste/aug/scale_heuristic.py` | depth-free scale fallback | NEW |
| `xpaste/aug/style_match.py` | Lab hist + selection + post-match | NEW |
| `xpaste/aug/build_augmented.py` | top-level offline driver | NEW |
| `xpaste/aug/visualize.py` | bbox sanity 시각화 | NEW |

## Experiment matrix

5 exps × YOLO11 (n/s/m) × 3 seeds = **45 runs**.

| Exp | Train data | Ablate |
|-----|-----------|--------|
| A | Real only | baseline |
| B | Real + GT crop random paste (paste_mode=real_random) | SD 풀 자체 가치 |
| C | Real + SD pool + random (paste_mode=pool_random) | naive X-Paste |
| D | Real + SD pool + scene-aware uniform (paste_mode=scene_uniform) | scene-awareness |
| **E** | **Real + SD pool + scene-aware + inverse-freq + style-match + post Lab match (paste_mode=style_full, full)** | **proposal** |
| F (sweep) | E with `--temperature {0.5, 2.0}` | inverse-freq 효과 검증 |

Eval: per-class AP50 on real test (197 imgs), 3 seeds mean ± std, paired bootstrap test for D vs E.

## Commands (server)

```bash
cd /workspace/XPaste
DATA_RAW=/workspace/datasets/Custom_Object_Detection_Military_v1
DATA=/workspace/XPaste/data/military_v1
POOL=/workspace/XPaste/output/pool_v1
CACHE=/workspace/XPaste/cache
mkdir -p "$DATA" "$POOL" "$CACHE"

# 1. 4-class 데이터 정리
python tools/remap_roboflow_labels.py --scheme four_class \
  --input_root "$DATA_RAW" --output_root "$DATA/real"

# 2. SD 풀 → CLIPSeg → margin 필터
python generation/gen_pose_instances.py \
  --poses_yaml configs/instance_poses_4cls.yaml --out "$POOL/raw" \
  --n_per_pose 30 --image_size 512 --steps 30 --guidance 7.5
python generation/segment_pose_hf.py --in "$POOL/raw" --out "$POOL/rgba"
python tools/filter_pool_by_clip_margin.py --in "$POOL/rgba" --out "$POOL/rgba_filtered" \
  --margin 0.10 --pairs "Soldier:civilian persons:soldier_uniform"

# 3. 분포 / 풀 캐시
python -m xpaste.aug.distribution --labels_dir "$DATA/real/train/labels" --out "$CACHE/dist_v1.json"
python -m xpaste.aug.style_match --pool_dir "$POOL/rgba_filtered" --out "$CACHE/pool_index_v1.npz"

# 4. 실험별 augmented set 생성 (B/C/D/E)
for MODE in real_random pool_random scene_uniform style_full; do
  case $MODE in real_random) E=B ;; pool_random) E=C ;; scene_uniform) E=D ;; style_full) E=E ;; esac
  python -m xpaste.aug.build_augmented \
    --host_root "$DATA/real/train" \
    --pool_dir "$POOL/rgba_filtered" --pool_index "$CACHE/pool_index_v1.npz" \
    --hist "$CACHE/dist_v1.json" \
    --paste_mode $MODE --pastes_per_image 3 --temperature 1.0 \
    --out_root "$DATA/aug_${E}/train" --seed 0
done

# 5. 시각화 sanity check
python -m xpaste.aug.visualize \
  --img "$DATA/aug_E/train/images/$(ls $DATA/aug_E/train/images | head -1)" \
  --label "$DATA/aug_E/train/labels/$(ls $DATA/aug_E/train/labels | head -1)" \
  --out viz/aug_E_check.png

# 6. 학습 행렬 (5 exp x 3 model x 3 seed = 45 runs)
bash tools/run_yolo_matrix.sh
```

## Reused from Original X-Paste

- `generation/text2im.py` — SD 1.5 wrapper
- `xpaste/data/transforms/custom_cp_method.py` — alpha/poisson blending
- `xpaste/data/transforms/possion_blending.py` — Poisson editing
- `segment_methods/{reseg,clean_pool,segment_pose_hf}.py` — instance seg/filter

## Deprecated (kept for reference, not in active path)

- `generation/gen_scenario_prompts.py` — LLM scenario prompt expansion
- `generation/gen_singleshot_scenes.py` — SDXL background generation
- `generation/filter_backgrounds.py` — DETR leak filter
- `generation/compose_scene.py` — replaced by `xpaste/aug/build_augmented.py`
- `configs/scenarios.yaml` — 10 scenario specs (no longer used)
- `configs/military.yaml` — 3-class config (replaced by `military_4cls.yaml`)

## Plan File

Full design + experiment matrix: `/Users/kyu216/.claude/plans/pure-stargazing-octopus.md`
