# Pipeline Tour 시각화 (서버 실행 가이드)

실제 host 이미지 1장이 분포-인식 Copy-Paste 파이프라인을 통과하며 처음부터 끝까지
변해가는 과정을 영상(mp4)으로 만든다. **실제 `xpaste/aug` 코드**(SegFormer, inverse-freq,
scale heuristic, Lab χ², L-shift, alpha-blend)를 그대로 호출한다.

## 0. 환경 (RTX A5000 / CUDA 11.7)
```bash
conda create -n viz python=3.10 -y && conda activate viz
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install transformers scikit-image pillow matplotlib numpy
sudo apt-get install -y fonts-nanum ffmpeg     # 한글 폰트 + 인코더
```
GPU는 1번이 한가하면 `--device cuda:1` 권장(0번은 학습 중).

## 1. (선택) 실제 증강 실행 → 좋은 예시 고르기
이미 aug_E를 만들었다면 건너뛴다. 새로 만들 때:
```bash
python -m xpaste.aug.build_augmented \
  --host_root data/military_v1/real/train \
  --pool_index cache/pool_index_v1.npz --hist cache/dist_v1.json \
  --paste_mode style_full --temperature 2.0 --pastes_per_image 3 \
  --save_meta --out_root data/military_v1/aug_E/train --seed 0
```
`aug_E/train/images`를 눈으로 보고 **합성이 자연스러운 이미지 1장**을 고른다(파일명 기억).
그 paste 정보는 `aug_E/train/meta.jsonl`에 들어있다.

## 2. 그 예시로 단계별 산출물 캡처
고른 이미지를 **그대로 재현**(meta 사용):
```bash
python tools/viz/capture_stages.py \
  --host_root data/military_v1/real/train \
  --host <고른_파일명.jpg> \
  --from_meta data/military_v1/aug_E/train/meta.jsonl \
  --pool_index cache/pool_index_v1.npz --hist cache/dist_v1.json \
  --out runs/viz/demo_stages --device cuda:1
```
또는 자동 선택 + 새 paste 샘플(메타 없이):
```bash
python tools/viz/capture_stages.py \
  --host_root data/military_v1/real/train \
  --pool_index cache/pool_index_v1.npz --hist cache/dist_v1.json \
  --temperature 2.0 --out runs/viz/demo_stages --device cuda:1
# SD 풀이 없으면 --demo_pool (실 GT crop으로 대체)
```

## 3. 영상 렌더
```bash
python tools/viz/render_example_tour.py \
  --cap runs/viz/demo_stages --out runs/viz/pipeline_tour.mp4
# 폰트 자동탐색 실패 시: --font /usr/share/fonts/truetype/nanum/NanumGothic.ttf
```
산출물: `runs/viz/pipeline_tour.mp4` (1920×1080, 30fps, ~23s)

## 흐름
intro → ① host → ② SegFormer region → ③ inverse-freq target → ④ scale+placement bbox
→ ⑤ Lab χ² style-match → ⑥ fly-in → ⑦ alpha-blend → ⑧ L-channel 보정 → ⑨ 결과 → outro

## 조정
- 구간 길이/캡션: `render_example_tour.py`의 `SEGS`, `CAPS`
- 예시 클래스 선호: `capture_stages.py --prefer_cls military_vehicle`
- host 자동선택 기준: `pick_host()` 점수식
