#!/usr/bin/env bash
# Run `yolo detect val` on every trained run in $RUNS for both OOD sets.
# Writes results to $RUNS/ood_{a,b}/<run_name>/results.csv.
#
# Env vars:
#   RUNS    runs root (default /workspace/XPaste/runs/military_v1)
#   DEVICE  CUDA device passed to yolo (default 0)
#   OOD     which OOD sets to evaluate (default "a b")
#   EXPS    exp letters to evaluate (default "A B C D E")

set -e

RUNS=${RUNS:-/workspace/XPaste/runs/military_v1}
DEVICE=${DEVICE:-0}
OOD=${OOD:-"a b"}
EXPS=${EXPS:-"A B C D E"}

for ood in $OOD; do
  DATA="configs/military_4cls_ood_${ood}.yaml"
  if [ ! -f "$DATA" ]; then
    echo "[skip] $DATA missing"
    continue
  fi
  for exp in $EXPS; do
    for d in "$RUNS"/${exp}_yolo11*_s[0-9]; do
      [ -d "$d" ] || continue
      name=$(basename "$d")
      best="$d/weights/best.pt"
      if [ ! -f "$best" ]; then
        echo "[skip] $name (no best.pt)"
        continue
      fi
      echo "[eval] OOD-${ood}  $name  device=$DEVICE"
      yolo detect val data="$DATA" model="$best" device="$DEVICE" \
        project="$RUNS/ood_${ood}" name="$name" exist_ok=True save_json=False
    done
  done
done
echo "[done] OOD eval"
