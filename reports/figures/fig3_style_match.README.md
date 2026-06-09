# Fig 3 — Style-matched instance selection (정성 비교)

> **상태**: PLACEHOLDER — 본 figure 의 실제 PNG 는 서버 `/workspace/XPaste` 환경에서 생성해야 함.
> 본 README 는 caption 초안과 재현 절차를 담아 두기 위한 문서.

## Caption (논문에 들어갈 문구 초안)

> **Fig. 3.** Style-matched instance selection 의 효과. 동일 host 이미지에 대해 (좌) Lab χ² 정렬 없이 무작위 선택된 SD 1.5 instance 를 paste한 결과와 (우) Lab histogram χ² 거리 기반 top-k 선택 + post Lab L-channel matching (cap ≤ 20) 을 적용한 결과. 우측이 host 의 색온도·조명과 자연스럽게 융합됨을 확인.

## 만드는 절차

1. host 한 장 선택
   ```bash
   HOST=$(ls /workspace/XPaste/data/military_v1/real/train/images | head -1)
   ```
2. (좌) random 선택 paste
   ```bash
   python -m xpaste.aug.build_augmented \
     --host_root /workspace/XPaste/data/military_v1/real/train \
     --pool_dir  /workspace/XPaste/output/pool_v1/rgba_filtered \
     --pool_index /workspace/XPaste/cache/pool_index_v1.npz \
     --hist /workspace/XPaste/cache/dist_v1.json \
     --paste_mode pool_random --pastes_per_image 3 --seed 0 \
     --filter_host "$HOST" \
     --out_root /tmp/fig3_random/train
   ```
3. (우) style_full + L-shift paste
   ```bash
   python -m xpaste.aug.build_augmented \
     ... --paste_mode style_full --temperature 2.0 --seed 0 \
     --filter_host "$HOST" --out_root /tmp/fig3_styled/train
   ```
   (※ 현재 build_augmented 에 `--filter_host` 가 없으면 임시 host 디렉토리를 만들어 이미지 1장만 두고 실행)
4. 두 결과 이미지를 horizontal concat → `fig3_style_match.png` 로 저장
   ```python
   from PIL import Image
   a = Image.open("/tmp/fig3_random/train/images/host.jpg")
   b = Image.open("/tmp/fig3_styled/train/images/host.jpg")
   w, h = a.size
   out = Image.new("RGB", (w*2 + 20, h), "white")
   out.paste(a, (0, 0)); out.paste(b, (w + 20, 0))
   out.save("reports/figures/fig3_style_match.png")
   ```

## 본문 인용 위치

`reports/paper_draft.md` §3.3 Style-matched Instance Selection.
