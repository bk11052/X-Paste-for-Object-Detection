#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/XPaste}"
CKPT_DIR="${ROOT}/segment_methods/checkpoints"

mkdir -p \
  "${CKPT_DIR}/clipseg/weights" \
  "${CKPT_DIR}/clipseg/matteformer" \
  "${CKPT_DIR}/u2net" \
  "${CKPT_DIR}/ufo" \
  "${CKPT_DIR}/selfreformer"

if ! command -v gdown >/dev/null 2>&1; then
  python -m pip install gdown
fi

echo "[1/5] CLIPSeg weights"
wget -O "${CKPT_DIR}/clipseg/weights/rd64-uni.pth" \
  https://github.com/timojl/clipseg/raw/master/weights/rd64-uni.pth

echo "[2/5] MatteFormer weights"
gdown --fuzzy "https://drive.google.com/file/d/1AU7uM1dtYjEhtOa_9OGfoQUE-tmW9mX5/view?usp=sharing" \
  -O "${CKPT_DIR}/clipseg/matteformer/best_model.pth"

echo "[3/5] U2Net weights"
gdown --fuzzy "https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing" \
  -O "${CKPT_DIR}/u2net/u2net.pth"

echo "[4/5] UFO weights"
gdown --fuzzy "https://drive.google.com/file/d/1ZFJwxBFTekAAxGuDMoafP4slTS_dBe3O/view?usp=sharing" \
  -O "${CKPT_DIR}/ufo/image_best.pth"

echo "[5/5] SelfReformer weights"
gdown --fuzzy "https://drive.google.com/file/d/19kO-IjZS56rIDTfscABhErTO-SP7oP4L/view?usp=sharing" \
  -O "${CKPT_DIR}/selfreformer/best_DUTS-TE.pt"

echo
echo "Done. Downloaded checkpoints under ${CKPT_DIR}"
