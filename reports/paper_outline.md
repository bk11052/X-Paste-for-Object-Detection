# Paper Outline — Distribution-aware Generative Copy-Paste for Military Object Detection

**대상 학회**: 한국군사과학기술학회 (KIMST)
**분량**: 2 페이지 이내 (KIMST 단편 기준)
**언어**: Abstract 영어, 본문 한글
**저자**: (TBD)

> 본 문서는 **논문 본문이 아닌 흐름/구성 outline** 입니다. 각 섹션의 핵심 메시지·근거 데이터·들어갈 figure 위치를 정리했습니다. 이후 이 outline 을 바탕으로 정식 본문을 채워 넣을 예정입니다.

---

## 0. 제목 & Abstract (English)

**Title (안)**
- *Distribution-aware Generative Copy-Paste Augmentation for Military Object Detection*
- (sub) *: Inverse-frequency Sampling and Style-matched Instance Selection*

**Abstract 핵심 메시지 (영어, ~150 단어)**
1. Military object detection 데이터는 (a) 샘플 수가 적고, (b) 클래스/스케일/위치 분포가 매우 편향되며, (c) 도메인 다양성이 부족.
2. 기존 Copy-Paste 계열은 instance 를 *균일하게* paste 하기 때문에 host 데이터의 long-tail 편향을 그대로 따라가는 한계.
3. 본 연구는 SD 1.5 로 합성한 instance 풀을 (class × scale × location) joint histogram 의 **inverse-frequency** 로 샘플링하고, host crop 과 Lab-histogram 으로 **style-matched** 인스턴스를 선택하여 paste.
4. 4-class military dataset (1,934 imgs) + YOLO11 (n/s/m × 3 seeds) 에서 **in-distribution mAP50 +2.3pt (vs baseline)**, **OOD-B (COCO civilian) mAP50 +3.3pt** 향상. **합성 데이터로 학습한 검출기가 in-distribution 성능을 잃지 않으면서 OOD 일반화를 개선** 한다는 것을 실증.

---

## 1. 서론 (Introduction)  — ~½ 단

**핵심 흐름**
1. **문제 제기**: 군용 객체 검출은 안전·전술 영향이 크지만, 데이터 수집이 어려워 long-tail · 적은 샘플 문제 심각.
2. **기존 접근의 한계**:
   - 실 GT crop 을 random paste → 다양성 부족 (Exp B 가 입증).
   - LLM/Diffusion 으로 *배경* 합성 → 본 연구 사전 실험에서 synth-only mAP50 < 0.01 로 실패 (CLAUDE.md 의 deprecated 경로). 즉 **배경 합성보다 instance 합성 + 실 배경에 paste 하는 augmentation 방향이 더 안정적**.
   - X-Paste(ICML 2023) 류는 instance 를 *균일* paste → host 데이터의 분포 편향을 그대로 학습.
3. **본 연구 기여 (3가지)**:
   - (i) Real train GT 의 **(class × scale × cx × cy) 4-D joint 히스토그램** 으로부터 inverse-frequency 샘플링 → tail bin 강조.
   - (ii) host crop 과 후보 인스턴스의 **Lab histogram χ²** 으로 style-matched 선택 + post-paste L-channel 보정 → 합성 instance 의 색감 mismatch 완화.
   - (iii) Depth-free **3-단 fallback scale heuristic** (same-class GT anchor → cy 회귀 → 분포 샘플) → depth 추정 실패에 robust.
4. **결과 요약**: in-dist 회복 + OOD-B(다양 배경) 일반화 동시 달성.

**Figure 후보 (서론용)**
- 없음 — 2페이지 제약상 서론은 짧게, figure 는 method/실험 섹션에 집중.

---

## 2. 관련 연구 (Related Work) — 4~5 줄로 압축

KIMST 단편에서는 보통 별도 섹션 없이 서론에 녹임. 단, 짧게 한 문단 정도로 다음을 언급:

- **Copy-Paste**: Ghiasi et al. (2021), X-Paste (ICML 2023) — instance level paste 의 효과.
- **Synthetic data with diffusion**: Stable Diffusion 으로 detection 학습 데이터 합성 시도 (DiffuMask 등), 그러나 **distribution-mismatch** 문제 미해결.
- **Long-tail detection**: re-weighting, re-sampling 계열은 분류기 수준에서만 다룸. 본 연구는 **데이터 생성 단계에서 분포를 직접 제어**.

> Figure 없음. 인라인 인용 한 문단.

---

## 3. 제안 방법 (Proposed Method) — Page 1 후반 ~ Page 2 초반

### 3.1 전체 파이프라인 개요

**텍스트 (3~4 줄)**: "real train 라벨로부터 4-D 분포를 추정 → SD 1.5 instance 풀에서 inverse-freq 샘플링 → host scene 분석 후 ground/road region 에 style-matched paste" 라는 한 문장 요약 + 그림 참조.

> **【Figure 1】 전체 파이프라인 (★ 필수, 1단 widescreen)**
> 좌→우 5 블록 흐름도:
> 1. Real train labels → joint hist (4×4×4×4=256 cells) 시각 (작은 heatmap)
> 2. Inverse-freq sampler (T=2.0)
> 3. SD 1.5 instance pool (썸네일 4 컷 × 4클래스) + Lab-style match
> 4. Host SegFormer (원본 + ground/road mask 오버레이)
> 5. 최종 paste 결과 (bbox 표시)
> *목적*: 독자가 한 눈에 method 의 입출력 흐름을 잡게 함.

### 3.2 Distribution-aware Inverse-Frequency Sampling

- (class, scale_bin, cx_bin, cy_bin) 4-D 히스토그램 \(P(c,s,x,y)\) 를 train GT 로부터 추정.
- 샘플링 가중치 \(w(c,s,x,y) \propto (P + \epsilon)^{-1/T}\), \(T=2.0\).
- T=1.0 vs T=2.0 ablation 결과: T=1.0 은 tail 과도 강조로 in-dist 손해, T=2.0 이 균형점.

> **【Figure 2】 분포 비교 (★ 필수, 2단 가로 분할)**
> 좌: per-class share — Real GT vs B/C (uniform from pool) vs D (scene uniform) vs **E (proposed)** 4 막대그래프.
> 우: scale bin 분포 (bin0~bin3) — 같은 4 모드.
> *근거 데이터 (aug_summary_v3.md)*:
> - Per-class: E 의 Soldier share **28.8%** (D 의 19.5% 대비 **+9.3pp**, GT 26.0% 보다도 약간 위 → tail 보강)
> - Scale: E 의 (bin0+bin3) **46.6%** (D 의 84.7% 가 bin0 에 몰린 것과 대비 → 작은+큰 객체 양 끝 oversampling)
> *메시지*: "uniform 대비 inverse-freq 가 분포의 어느 부분을 채우는지" 를 시각적으로 입증.

### 3.3 Style-matched Instance Selection

- 후보 N=8 개 인스턴스 vs host crop 의 Lab histogram **χ² distance** 계산 → top-k=8 중 가중 샘플.
- Paste 후 host crop L-channel mean shift (cap ≤20) 적용.

> **【Figure 3】 Style-match 정성 효과 (★ 권장, 작은 2 컷)**
> 좌: random pool 선택 → 색감 mismatch (예: 어두운 host 에 밝은 instance 가 paste 된 케이스)
> 우: style-matched + L-channel 보정 → 자연스러운 paste
> *목적*: 정량 ablation 만으로 잡기 어려운 시각 품질 효과 호소.

### 3.4 Depth-free Scale Heuristic

3-단 fallback:
1. **gt_anchor**: 같은 host 에 동일 클래스 GT 가 있으면 그 면적 비율로 스케일 결정.
2. **cy_regression**: per-class \(cy \to h\) 선형 회귀 (train GT 로 사전 학습).
3. **target_scale**: sampler 가 정한 scale bin 중점값으로 fallback.

aug_summary_v3.md 근거: E 모드에서 **gt_anchor 58.0% / cy_reg 13.3% / target_scale 28.7%** → primary path 인 GT-anchor 가 과반.

> **【Figure 없음】** 텍스트 + 인라인 분율로 처리 (지면 절약).

---

## 4. 실험 (Experiments) — Page 2 본체

### 4.1 실험 설정

- **데이터셋**: Roboflow `Custom Object Detection -Military-` (1,934 imgs, 4-class). Roboflow 기본 split 이 leakage 가 있어 **scene-disjoint split** 으로 재분할 (train 1,352 / val 385 / test 197).
- **OOD test**: OOD-A (Roboflow military, 2,636 imgs, 도메인 內 다른 분포), OOD-B (COCO val2017 subset 500 imgs, civilian 도메인).
- **검출기**: YOLO11 n/s/m × 3 seeds (총 45 runs = 5 exp × 3 model × 3 seed).
- **비교 모드**:
  - A: Real only (baseline)
  - B: Real + GT crop random paste (SD 풀 자체 가치 검증)
  - C: Real + SD pool, random paste (naive X-Paste)
  - D: Real + SD pool + scene-aware uniform (scene-awareness 만)
  - **E (제안)**: Real + SD pool + scene-aware + inverse-freq (T=2.0) + style-match
- **Augmentation 통계** (aug_summary_v3.md): C/D/E 모두 같은 SD 풀 사용. host accept rate D=E=53.1%, B=C=100%. paste 수 B=C=1,755, D=973, E=910 (E 가 가장 selective).

> **【Figure 4】 4-mode paste 정성 비교 (★ 필수, 2×2 grid)**
> 동일 host image 에 대해 B/C/D/E 각각의 paste 결과 + bbox 오버레이.
> *목적*: 증강 결과의 시각 차이 (특히 D vs E 의 region 이질성, E 의 색감 통합) 를 한 눈에.
> *주의*: 같은 host 를 사용해야 비교가 정당. 실험 끝난 train 셋에서 한 장을 골라 4 모드 동시 재현.

### 4.2 정량 결과 — In-distribution + OOD

> **【Table 1】 통합 정량 결과 (★ 필수)**
> 모델별 (n/s/m) × Exp (A/B/C/D/E) × Split (ID / OOD-A / OOD-B) × mAP50.
> seed 평균 (3 seeds). 본문 분량상 **mAP50 만** 제시, mAP50-95 는 부록 또는 Appendix 표.
>
> 핵심 셀 (Final_Experimental_Results.md 발췌):
> - **ID m**: A 0.902 / B 0.910 / C 0.912 / D 0.918 / **E 0.925** (▲+2.3pt vs A, ▲+0.7pt vs D)
> - **OOD-A m**: A 0.133 / B 0.134 / C **0.214** / D 0.201 / E 0.182 — *E 가 1등은 아님, naive SD (C) 가 우세*
> - **OOD-B m**: A 0.347 / B 0.347 / C 0.355 / D 0.344 / **E 0.380** (▲+3.3pt vs A, ▲+3.6pt vs D)

**해석 (본문 4~5 줄)**:
1. **In-dist**: E 가 모든 model size 에서 1위. catastrophic forgetting 없음.
2. **OOD-B (civilian, COCO)**: E 가 모든 model size 에서 1위. **distribution-aware 가 도메인 격차가 큰 OOD 에서 가장 효과**.
3. **OOD-A (military 다른 분포)**: 의외로 naive SD (C) 가 큰 모델에서 우세. 가설: OOD-A 도메인은 host 와 *유사*해서 inverse-freq 의 tail 강조가 오히려 over-fit 을 깸. 이 점은 paper 에서 **솔직히 trade-off 로 기술** + 향후 과제로 제시.

### 4.3 Ablation — 어떤 모듈이 기여했는가?

C → D → E 의 cumulative ablation 으로 자연스럽게 ablation 이 됨:
- C → D: scene-awareness (region match) → in-dist +0.6pt
- D → E (m): inverse-freq + style-match → in-dist +0.7pt, OOD-B +3.6pt

> **【Figure 5】 ablation bar (★ 권장, 작은 그래프)**
> X 축: A/B/C/D/E. Y 축: mAP50. 3 개 막대 (ID, OOD-A, OOD-B) 그룹화. yolo11m seed-mean.
> *목적*: 표 1 의 핵심 메시지를 한 컷에 압축.

---

## 5. 결론 및 향후 과제 (Conclusion) — 4~5 줄

- 군용 객체 검출의 **데이터 분포 편향** 문제를 **합성 데이터 분포 제어** 라는 새로운 각도로 접근.
- (class × scale × location) 4-D inverse-freq + style-match 로 in-dist 손해 없이 OOD 일반화 +3pt 달성.
- **한계**: OOD-A (가까운 도메인) 에서는 naive 대비 우위 없음. 향후 host-domain 유사도에 따라 T 를 동적으로 조정하는 adaptive sampler 가 필요.

---

## 부록 / 보조 자료 (필요시 supplementary)

- **【Figure S1】** SD 1.5 인스턴스 풀 샘플 4×4 grid (4 클래스 × 4 자세). 합성 품질 시각.
- **【Figure S2】** Host accept/reject 통계 (insufficient_paste_region 이 D/E 거절의 ~50% 설명).
- **【Table S1】** Per-class AP50 breakdown (Soldier / civilian_vehicle / military_vehicle / persons) × A/B/C/D/E × n/s/m.
- **【Table S2】** mAP50-95 통합 결과 (본문에 빠진 metric).

---

## Figure / Table 우선순위 정리

| ID | 종류 | 위치 | 우선순위 | 설명 |
|---|---|---|---|---|
| Fig 1 | Pipeline overview | §3.1 | ★★★ | 5-block 흐름도, 1단 widescreen |
| Fig 2 | 분포 비교 (class/scale) | §3.2 | ★★★ | E vs D vs GT 막대 (inverse-freq 핵심 증거) |
| Fig 3 | Style-match 정성 | §3.3 | ★★ | 2 컷 before/after |
| Fig 4 | 4-mode paste 정성 비교 | §4.1 | ★★★ | 2×2 grid, 같은 host |
| Fig 5 | Ablation bar | §4.3 | ★★ | A~E × ID/OOD-A/OOD-B |
| Tab 1 | 통합 mAP50 결과 | §4.2 | ★★★ | 모든 model size × split |
| Fig S1 | SD pool 샘플 | App | ★ | 합성 품질 |
| Tab S1 | Per-class AP50 | App | ★ | 디테일 |

> **본문 권장 figure 수**: 4 개 (Fig 1, 2, 4, 5) + Table 1.
> Fig 3 (style-match 정성) 은 지면 여유시. 너무 많으면 지면 압박.

---

## 다음 단계 제안

1. 이 outline 의 **메시지 라인** 동의 여부 확인 (특히 OOD-A 에서 E 가 1등 아님을 솔직히 기술하는 방향).
2. Figure 1 (pipeline) 의 도식안 스케치.
3. Figure 2 (분포 비교) 데이터는 이미 `aug_summary_v3.md` 에 있음 → matplotlib 으로 즉시 그릴 수 있음.
4. Figure 4 (4-mode paste) 는 같은 host 로 4 mode 재현 필요 → 작은 스크립트 한 번 돌려야 함.
5. 메시지 동의 시 **본문 본격 작성** 진행.
