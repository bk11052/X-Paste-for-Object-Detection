# Fig 4 — 4-mode paste 정성 비교 (B / C / D / E)

> **상태**: PLACEHOLDER — 본 figure 의 실제 PNG 는 서버 `/workspace/XPaste` 환경에서 생성해야 함.

## Caption (논문에 들어갈 문구 초안)

> **Fig. 4.** 동일 host 이미지에 대한 4 가지 augmentation 모드의 paste 결과. (a) Mode B: 실 GT crop 무작위 paste. (b) Mode C: SD 1.5 풀에서 무작위 선택. (c) Mode D: scene-aware uniform paste (SegFormer ground/road region 위에만 placement). (d) Mode E (Ours): scene-aware + inverse-frequency sampler (T=2.0) + style-matched 선택 + L-channel post matching. (d) 가 (b)~(c) 대비 지면·전경 정합성과 색감 일관성이 개선되었음을 확인.

## 만드는 절차

1. host 한 장 고정
   ```bash
   HOST=$(ls /workspace/XPaste/data/military_v1/real/train/images | head -1)
   echo "$HOST"
   ```
2. 이미 build 한 aug_{B,C,D,E} 디렉토리에서 같은 host 의 결과 이미지를 추출
   ```bash
   for x in B C D E; do
     cp "/workspace/XPaste/data/military_v1/aug_${x}/train/images/${HOST}" "/tmp/fig4_${x}.jpg"
     # bbox 오버레이
     python -m xpaste.aug.visualize \
       --img   "/tmp/fig4_${x}.jpg" \
       --label "/workspace/XPaste/data/military_v1/aug_${x}/train/labels/${HOST%.*}.txt" \
       --out   "/tmp/fig4_${x}_viz.jpg"
   done
   ```
3. 2×2 grid 합성
   ```python
   from PIL import Image, ImageDraw, ImageFont
   tags = [("B", "Real Paste"), ("C", "Naive SD"), ("D", "Scene-aware"), ("E", "Ours")]
   imgs = [Image.open(f"/tmp/fig4_{t}_viz.jpg") for t, _ in tags]
   w, h = imgs[0].size
   pad = 16
   out = Image.new("RGB", (w*2 + pad*3, h*2 + pad*3 + 40), "white")
   for i, im in enumerate(imgs):
       r, c = i // 2, i % 2
       x, y = pad + c*(w+pad), pad + 40 + r*(h+pad)
       out.paste(im, (x, y))
       d = ImageDraw.Draw(out)
       d.text((x, y - 22), f"({tags[i][0]}) {tags[i][1]}", fill="black")
   out.save("reports/figures/fig4_4mode_paste.png")
   ```

## 주의

- 4 모드가 같은 host 이미지를 공유하려면 build_augmented 시 동일 seed + 동일 host 가 모두 paste 대상이 되어야 함.
- D/E 는 host accept rate 53.1% 이므로 reject 된 host 일 경우 다른 host 로 교체하여 재시도.

## 본문 인용 위치

`reports/paper_draft.md` §4.3 정성 결과.
