#!/usr/bin/env bash
# YOLO11 experiment matrix driver: 5 exp x 3 model size x 3 seed = 45 runs.
# Each experiment uses its own train split under data/military_v1/aug_<X>/train,
# while val and test always come from data/military_v1/real/{valid,test}.
#
# Prerequisites:
#   - data/military_v1/real/{train,valid,test} populated (remap_roboflow_labels.py)
#   - data/military_v1/aug_{B,C,D,E}/train populated (build_augmented.py per mode)
#   - configs/military_4cls.yaml with default `train: real/train/images`
#
# Usage:
#   bash tools/run_yolo_matrix.sh                # full matrix
#   EXPS="A E" MODELS="yolo11s" SEEDS="0" bash tools/run_yolo_matrix.sh    # subset
set -euo pipefail

ROOT="${ROOT:-/workspace/XPaste}"
DATA="${DATA:-${ROOT}/data/military_v1}"
RUNS="${RUNS:-${ROOT}/runs/military_v1}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
EXPS="${EXPS:-A B C D E}"
MODELS="${MODELS:-yolo11n yolo11s yolo11m}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-0}"

# Map exp letter -> train images path (relative to data.yaml `path`)
declare -A TRAIN_PATH
TRAIN_PATH[A]="real/train/images"
TRAIN_PATH[B]="aug_B/train/images"
TRAIN_PATH[C]="aug_C/train/images"
TRAIN_PATH[D]="aug_D/train/images"
TRAIN_PATH[E]="aug_E/train/images"

mkdir -p "${RUNS}"
TMPL="${ROOT}/configs/military_4cls.yaml"

for EXP in $EXPS; do
  TRAIN="${TRAIN_PATH[$EXP]:-}"
  if [ -z "$TRAIN" ]; then
    echo "[skip] unknown exp $EXP"; continue
  fi
  if [ ! -d "${DATA}/${TRAIN}" ]; then
    echo "[skip] missing ${DATA}/${TRAIN}"; continue
  fi

  CFG="${RUNS}/cfg_${EXP}.yaml"
  # Generate per-experiment data.yaml by swapping the `train:` line.
  awk -v t="$TRAIN" '/^train:/{print "train: " t; next}1' "$TMPL" > "$CFG"

  for MODEL in $MODELS; do
    for SEED in $SEEDS; do
      NAME="${EXP}_${MODEL}_s${SEED}"
      OUT="${RUNS}/${NAME}"
      if [ -d "$OUT" ]; then
        echo "[skip] $OUT exists"; continue
      fi
      echo "[run]  $NAME  cfg=$CFG"
      yolo detect train \
        data="$CFG" \
        model="${MODEL}.pt" \
        epochs="$EPOCHS" \
        imgsz="$IMGSZ" \
        batch="$BATCH" \
        seed="$SEED" \
        device="$DEVICE" \
        project="${RUNS}" \
        name="$NAME" \
        exist_ok=False
    done
  done
done

echo "[done] all runs under ${RUNS}"
