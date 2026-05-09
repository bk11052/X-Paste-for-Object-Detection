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
| `tools/check_split_leakage.py` | prefix-overlap leakage 진단 | NEW |
| `tools/scene_disjoint_split.py` | raw Roboflow → scene-disjoint split | NEW |
| `tools/filter_pool_by_clip_margin.py` | R1 mitigation 풀 필터 | NEW |
| `tools/run_yolo_matrix.sh` | 실험 행렬 드라이버 | NEW |
| `tools/build_ood_military.py` | Roboflow military source → ood_a/ (flexible class remap) | NEW |
| `tools/build_coco_civilian_ood.py` | COCO val2017 subset → ood_b/ (persons + civilian_vehicle) | NEW |
| `tools/eval_on_ood.sh` | trained model × OOD-{a,b} `yolo val` 드라이버 | NEW |
| `configs/ood_military_mapping.yaml` | OOD-A 클래스 id 매핑 | NEW |
| `configs/military_4cls_ood_a.yaml` | OOD-A YOLO eval config | NEW |
| `configs/military_4cls_ood_b.yaml` | OOD-B YOLO eval config | NEW |
| `xpaste/aug/__init__.py` | aug 패키지 (CLASSES) | NEW |
| `xpaste/aug/distribution.py` | joint hist + inverse-freq sampler | NEW |
| `xpaste/aug/host_scene.py` | host SegFormer + accept/reject | NEW |
| `xpaste/aug/scale_heuristic.py` | depth-free scale fallback | NEW |
| `xpaste/aug/style_match.py` | Lab hist + selection + post-match | NEW |
| `xpaste/aug/build_augmented.py` | top-level offline driver (`--save_meta` JSONL provenance) | NEW |
| `xpaste/aug/visualize.py` | bbox sanity 시각화 | NEW |
| `tools/summarize_aug_distribution.py` | meta.jsonl → paper용 marginal/joint 분포 markdown | NEW |

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

## Commands (server, 4 GPUs)

호스트 `/home/gpuadmin/Gachon/kyu216/X-Paste/XPaste`. Docker 진입 후 `/workspace/XPaste`에서 작업. **항상 `tmux` 안에서 실행** (nohup wait가 셸 끊기면 풀림 — R9). 자세한 phase 의존도/시간 추정은 `/Users/kyu216/.claude/plans/pure-stargazing-octopus.md` 참조.

전체 흐름:
```
Phase 0 setup (10분)
  ↓
Phase 1A Exp A train (GPU 0,1,2)  ─┐
Phase 1B SD pool gen (GPU 3)      ─┴── 병렬 ~2.5h
  ↓
Phase 2 dist + pool index (5분)
  ↓
Phase 3 aug B/C/D/E build (GPU 0~3 병렬, 30~60분)
  ↓
Phase 4 Exp B/C/D/E train (GPU 0~3 병렬, ~10h)
  ↓
Phase 5 분석 (1h)
```

### Phase 0 — setup
```bash
# (호스트) docker 진입
cd /home/gpuadmin/Gachon/kyu216/X-Paste/XPaste
git checkout feat/dist-aware-paste-4cls && git pull
tmux new -s xpaste
docker run --gpus all -it --rm --shm-size=32g \
  -v $(pwd):/workspace/XPaste \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  xpaste bash

# (컨테이너 안)
cd /workspace/XPaste
pip install ultralytics "numpy<2" scikit-image open-clip-torch
python -c "import numpy, ultralytics, skimage, open_clip; print('deps ok', numpy.__version__)"

export DATA_RAW="/workspace/XPaste/data/Custom Object Detection -Military-.v1i.yolov8"
export DATA="/workspace/XPaste/data/military_v1"
export POOL="/workspace/XPaste/output/pool_v1"
export CACHE="/workspace/XPaste/cache"
export RUNS="/workspace/XPaste/runs/military_v1"
mkdir -p "$DATA" "$POOL" "$CACHE" "$RUNS" /workspace/XPaste/viz logs

# Roboflow 기본 split은 image-level random — scene-disjoint 보장 X.
# 첫 학습에서 yolo11n mAP50=0.913 (예상 0.55)가 나와 leakage 확정 → scene-disjoint 재분할.
python tools/check_split_leakage.py --raw_root "$DATA_RAW"
# train∩test / |test| > 5%면 재분할 진행:
python tools/scene_disjoint_split.py \
  --raw_root "$DATA_RAW" --out_root "$DATA/real_raw_new" \
  --ratios 0.70 0.20 0.10 --seed 0
python tools/remap_roboflow_labels.py --scheme four_class \
  --input_root "$DATA/real_raw_new" --output_root "$DATA/real"
rm -rf "$DATA/real_raw_new"

ls "$DATA/real/train/images" | wc -l   # ~1352
ls "$DATA/real/valid/images" | wc -l   # ~385
ls "$DATA/real/test/images"  | wc -l   # ~197
```

### Phase 1 — Exp A baseline + SD 풀 (병렬 launch)
```bash
# 1A: Exp A 9 runs (GPU 0/1/2 each → seed 0,1,2 순차)
DEVICE=0 EXPS=A MODELS=yolo11n SEEDS="0 1 2" \
  nohup bash tools/run_yolo_matrix.sh > logs/A_n.log 2>&1 &
DEVICE=1 EXPS=A MODELS=yolo11s SEEDS="0 1 2" \
  nohup bash tools/run_yolo_matrix.sh > logs/A_s.log 2>&1 &
DEVICE=2 EXPS=A MODELS=yolo11m SEEDS="0 1 2" \
  nohup bash tools/run_yolo_matrix.sh > logs/A_m.log 2>&1 &

# 1B: SD 풀 (GPU 3)
# 주의: gen_pose_instances.py는 --scenarios/--output_dir/--samples,
#       segment_pose_hf.py는 segment_methods/ 아래에 있고 --input_dir/--output_dir.
CUDA_VISIBLE_DEVICES=3 nohup bash -c '
  set -e
  python generation/gen_pose_instances.py \
    --scenarios configs/instance_poses_4cls.yaml \
    --output_dir "$POOL/raw" --samples 30 --image_size 512 --steps 30 --guidance 7.5
  python segment_methods/segment_pose_hf.py \
    --input_dir "$POOL/raw" --output_dir "$POOL/rgba"
  python tools/filter_pool_by_clip_margin.py \
    --in "$POOL/rgba" --out "$POOL/rgba_filtered" \
    --margin 0.10 --pairs "Soldier:civilian persons:soldier_uniform"
' > logs/sdpool.log 2>&1 &

# 모니터링: tail -f logs/A_*.log logs/sdpool.log
wait
echo "[Phase 1] complete"

# Sanity 확인
for M in yolo11n yolo11s yolo11m; do
  for S in 0 1 2; do
    F="$RUNS/A_${M}_s${S}/results.csv"
    [ -f "$F" ] && echo "A_${M}_s${S}: $(tail -1 $F | awk -F, '{print "mAP50="$8}')" || echo "A_${M}_s${S}: MISSING"
  done
done
ls "$POOL/rgba_filtered" | awk -F'__' '{print $1}' | sort | uniq -c
```

기대값: yolo11n mAP50 ~0.55, s ~0.62, m ~0.66 (±0.03). 풀 클래스당 ≥100 인스턴스.

### Phase 2 — 분포 + 풀 인덱스
```bash
python -m xpaste.aug.distribution \
  --labels_dir "$DATA/real/train/labels" --out "$CACHE/dist_v1.json"
python -m xpaste.aug.style_match \
  --pool_dir "$POOL/rgba_filtered" --out "$CACHE/pool_index_v1.npz"
```

### Phase 3 — aug 데이터셋 4종 (4-way 병렬)

`--save_meta` 가 핵심: 각 paste의 (class, scale_bin, cx_bin, cy_bin), scale_method, SD 풀 인스턴스 출처가 `<out_root>/meta.jsonl`에 기록됨. paper 본문 표 (sampler 효과 입증용) 필수.

```bash
# 기존 aug 디렉토리 백업 (이미 빌드된 게 있다면):
for x in B C D E; do
  [ -d "$DATA/aug_${x}" ] && mv "$DATA/aug_${x}" "$DATA/aug_${x}.bak.$(date +%s)"
done

build_aug() {
  local MODE=$1 EXP=$2 GPU=$3
  CUDA_VISIBLE_DEVICES=$GPU nohup python -m xpaste.aug.build_augmented \
    --host_root "$DATA/real/train" \
    --pool_dir "$POOL/rgba_filtered" --pool_index "$CACHE/pool_index_v1.npz" \
    --hist "$CACHE/dist_v1.json" \
    --paste_mode "$MODE" --pastes_per_image 3 --temperature 1.0 \
    --out_root "$DATA/aug_${EXP}/train" --save_meta --seed 0 \
    > "logs/aug_${EXP}.log" 2>&1 &
}
build_aug real_random   B 0
build_aug pool_random   C 1
build_aug scene_uniform D 2
build_aug style_full    E 3
wait

# 검증: meta.jsonl 라인 수 == 이미지 수
for x in B C D E; do
  N_IMG=$(ls "$DATA/aug_${x}/train/images" | wc -l)
  N_META=$(wc -l < "$DATA/aug_${x}/train/meta.jsonl")
  echo "aug_${x}: images=$N_IMG meta_lines=$N_META"
done

python -m xpaste.aug.visualize \
  --img  "$DATA/aug_E/train/images/$(ls $DATA/aug_E/train/images | head -1)" \
  --label "$DATA/aug_E/train/labels/$(ls $DATA/aug_E/train/labels | head -1)" \
  --out viz/aug_E_check.png
```

### Phase 3.5 — aug 분포 summary (paper 본문 표)

meta.jsonl 4개 → 모드 간 marginal/joint 분포 비교 markdown.

```bash
mkdir -p reports
python tools/summarize_aug_distribution.py \
  --aug_root "$DATA" --modes B C D E \
  --hist "$CACHE/dist_v1.json" \
  --out reports/aug_summary.md
cat reports/aug_summary.md
```

기대 결과:
- **per-class**: E의 underrepresented class (`military_vehicle`, GT 363개) 비율이 D 대비 +5pp 이상이면 inverse-freq 효과 입증.
- **scale bin**: E의 bin 0 + bin 3 (tail) 합이 D 대비 명확히 큼.
- **scale method**: E에서 `gt_anchor` 가 50%+ 면 fallback 사용 빈도 정상.

### Phase 4 — Exp B/C/D/E 학습 (36 runs, 4-way 병렬)
```bash
DEVICE=0 EXPS=B MODELS="yolo11n yolo11s yolo11m" SEEDS="0 1 2" \
  nohup bash tools/run_yolo_matrix.sh > logs/B_all.log 2>&1 &
DEVICE=1 EXPS=C MODELS="yolo11n yolo11s yolo11m" SEEDS="0 1 2" \
  nohup bash tools/run_yolo_matrix.sh > logs/C_all.log 2>&1 &
DEVICE=2 EXPS=D MODELS="yolo11n yolo11s yolo11m" SEEDS="0 1 2" \
  nohup bash tools/run_yolo_matrix.sh > logs/D_all.log 2>&1 &
DEVICE=3 EXPS=E MODELS="yolo11n yolo11s yolo11m" SEEDS="0 1 2" \
  nohup bash tools/run_yolo_matrix.sh > logs/E_all.log 2>&1 &
wait
```

### Phase 4.5 — OOD test 빌드 (Phase 4 도는 동안 병렬)
```bash
# OOD-A (military, Roboflow yolo-datasets-ymdve)
mkdir -p data/military_v1/ood_a_raw && cd data/military_v1/ood_a_raw
# Roboflow Universe → Download Dataset → YOLOv8 → curl 복사 (사용자 직접)
# 압축 해제 후 cd /workspace/XPaste
python tools/build_ood_military.py \
  --source_root data/military_v1/ood_a_raw \
  --mapping_yaml configs/ood_military_mapping.yaml \
  --output_root data/military_v1/ood_a

# OOD-B (civilian, COCO val2017 subset)
mkdir -p data/coco && cd data/coco
[ -d val2017 ]     || (wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip)
[ -d annotations ] || (wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip)
cd /workspace/XPaste
python tools/build_coco_civilian_ood.py \
  --coco_imgs data/coco/val2017 \
  --coco_ann  data/coco/annotations/instances_val2017.json \
  --output_root data/military_v1/ood_b \
  --max_images 500 --seed 0
```

### Phase 4.6 — OOD eval (Phase 4 train 끝난 뒤)
```bash
# 전체를 1 GPU에서 순차 (~3h):
DEVICE=0 RUNS="$RUNS" bash tools/eval_on_ood.sh > logs/ood_eval.log 2>&1 &

# 또는 4 GPU 분산 (각 EXP별로):
for i in 0 1 2 3; do
  EXP=$(echo "B C D E" | cut -d' ' -f$((i+1)))
  DEVICE=$i EXPS="$EXP" RUNS="$RUNS" \
    nohup bash tools/eval_on_ood.sh > "logs/ood_${EXP}.log" 2>&1 &
done
DEVICE=0 EXPS=A RUNS="$RUNS" nohup bash tools/eval_on_ood.sh > logs/ood_A.log 2>&1 &
wait
```

### Phase 5 — 결과 집계 (in-distribution + OOD-A + OOD-B)
```bash
python - <<'PY'
import pandas as pd, glob, os, re
RUNS = os.environ['RUNS']
PAT = re.compile(r'(\w)_yolo11(\w)_s(\d)')

def collect(root, label):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, '[A-E]_yolo11*_s*'))):
        name = os.path.basename(d)
        csv = os.path.join(d, 'results.csv')
        if not os.path.exists(csv): continue
        df = pd.read_csv(csv)
        last = df.iloc[-1]
        m = PAT.match(name)
        if not m: continue
        rows.append(dict(
            split=label, exp=m.group(1), model='yolo11'+m.group(2), seed=int(m.group(3)),
            mAP50=last.get('metrics/mAP50(B)', float('nan')),
            mAP=last.get('metrics/mAP50-95(B)', float('nan')),
        ))
    return rows

all_rows = []
all_rows += collect(RUNS, 'in_dist')
all_rows += collect(os.path.join(RUNS, 'ood_a'), 'ood_a')
all_rows += collect(os.path.join(RUNS, 'ood_b'), 'ood_b')

out = pd.DataFrame(all_rows)
print(out.groupby(['split','exp','model'])['mAP50'].agg(['mean','std']).unstack().round(3))
out.to_csv(os.path.join(RUNS, 'summary.csv'), index=False)
PY
```

핵심 비교:
- **in-dist** A vs E (+2.0 mAP50 이상이면 paper-worthy), D vs E (+0.8 이상이면 inverse-freq + style-match novelty 정당)
- **OOD-A** (Soldier + military_vehicle): E의 절대값보다 E vs A 상대 게인이 핵심. small-object generalization 증거.
- **OOD-B** (civilian_vehicle + persons): COCO 도메인 격차 큼. 모든 모델 낮을 수 있음, 상대 비교 위주.

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
