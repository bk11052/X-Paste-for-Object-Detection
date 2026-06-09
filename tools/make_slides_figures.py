# -*- coding: utf-8 -*-
"""Generate per-slide presentation visuals for the KIMST 2026 oral talk.

All numbers are taken from the paper / reports:
  - reports/Final_Experimental_Results.md (YOLO11-m ID & OOD mAP)
  - reports/aug_summary_v3.md (per-class / scale-bin paste share, scale method)

Outputs go to the visualization folder next to the paper PDF.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- Korean font ----
for cand in ["Apple SD Gothic Neo", "AppleGothic", "Nanum Gothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("/Users/kyu216/Univ/학부연구생/논문/02. MilitaryDataset/visualization")
OUT.mkdir(parents=True, exist_ok=True)

# Palette
C_GT   = "#8a8f99"
C_REAL = "#6c757d"
C_D    = "#4C9F70"
C_E    = "#D7263D"
C_BLUE = "#2F6DB5"
C_GREY = "#cfd3da"
BOX_BG = "#eef2f7"
BOX_ED = "#9fb3c8"


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p.name)


# ============================================================
# Slide 2 — class long-tail bar chart
# ============================================================
def slide02():
    classes = ["Soldier", "civilian_vehicle", "military_vehicle", "persons"]
    counts  = [577, 424, 363, 856]
    order = np.argsort(counts)[::-1]
    classes = [classes[i] for i in order]
    counts  = [counts[i] for i in order]
    colors  = [C_E if c == min(counts) else C_BLUE for c in counts]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=200)
    bars = ax.bar(classes, counts, color=colors, width=0.62, edgecolor="white")
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, c+12, str(c),
                ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.axhline(np.mean(counts), ls="--", color="#888", lw=1)
    ax.text(3.45, np.mean(counts)+8, f"평균 {np.mean(counts):.0f}",
            ha="right", color="#666", fontsize=10)
    ax.annotate("Tail (최소 클래스)", xy=(2, 363), xytext=(2.05, 560),
                fontsize=11, color=C_E, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_E, lw=1.6))
    ax.set_ylabel("Train GT 인스턴스 수", fontsize=12)
    ax.set_ylim(0, 980)
    ax.set_title("군용 데이터셋의 Long-tail 클래스 분포  (총 2,220 인스턴스)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.text(0.5, -0.22, "persons : military_vehicle ≈ 2.4 : 1  —  클래스 빈도 불균형",
            transform=ax.transAxes, ha="center", fontsize=10.5, color="#444")
    save(fig, "slide02_class_longtail.png")


# ============================================================
# Slide 3 — uniform sampling replicates bias (concept)
# ============================================================
def slide03():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=200)
    labels = ["S", "civ", "mil", "per"]
    real = np.array([26, 19, 16, 39])
    uni  = real + np.array([0, 1, 1, -2])   # uniform paste ~ keeps shape
    ours = np.array([29, 22, 23, 27])       # inverse-freq flattens

    def draw(ax, vals, title, color, tail_hi=False):
        cols = [color]*4
        if tail_hi:
            cols[2] = C_E
        ax.bar(labels, vals, color=cols, width=0.65, edgecolor="white")
        ax.set_ylim(0, 46)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.set_ylabel("share (%)", fontsize=9)

    draw(axes[0], real, "① 원본 GT 분포\n(편향 존재)", C_REAL)
    draw(axes[1], uni,  "② 기존 Copy-Paste\n(균일 샘플링 → 편향 복제)", C_D)
    draw(axes[2], ours, "③ 제안 (inverse-freq)\n(tail 보강 → 평탄화)", C_BLUE, tail_hi=True)

    # arrows between panels
    for x in (0.345, 0.655):
        fig.text(x, 0.5, "→", fontsize=26, color="#999", ha="center", va="center")
    fig.suptitle("균일 샘플링은 원본 편향을 그대로 학습한다  vs  분포 인지 보정",
                 fontsize=13, fontweight="bold", y=1.04)
    save(fig, "slide03_problem_concept.png")


# ============================================================
# Slide 5 — inverse-frequency: weight curve + scale-bin shift
# ============================================================
def slide05():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=200)

    # (a) weight curve w ∝ (P+eps)^(-1/T)
    P = np.linspace(0.005, 1.0, 200)
    eps = 1e-3
    for T, c, ls in [(0.5, "#9ecae1", "-"), (1.0, "#4292c6", "-"),
                     (2.0, C_E, "-"), (1e6, "#999", "--")]:
        w = (P + eps) ** (-1.0 / T)
        w = w / w.max()
        lab = "T→∞ (uniform)" if T > 100 else f"T = {T}"
        axL.plot(P, w, ls, color=c, lw=2.4 if T == 2.0 else 1.8, label=lab)
    axL.set_xlabel("cell 빈도  P(c, s, x, y)", fontsize=11)
    axL.set_ylabel("샘플링 가중치 (정규화)", fontsize=11)
    axL.set_title("(a) 역빈도 가중치  w ∝ (P+ε)$^{-1/T}$", fontsize=12, fontweight="bold")
    axL.legend(fontsize=9.5, loc="upper right")
    axL.grid(ls=":", alpha=0.5)
    axL.text(0.97, 0.06, "희소 cell일수록\n큰 가중치 → tail 보강",
             transform=axL.transAxes, ha="right", fontsize=9.5, color=C_E)

    # (b) scale-bin share D vs E (real data)
    bins = ["bin0\n(small)", "bin1", "bin2", "bin3\n(large)"]
    d = [82.5, 15.3, 2.2, 0.0]
    e = [24.5, 28.4, 25.1, 22.1]
    x = np.arange(4); w = 0.38
    axR.bar(x-w/2, d, w, label="Mode D (uniform)", color=C_D)
    axR.bar(x+w/2, e, w, label="Mode E (Ours)", color=C_E)
    axR.set_xticks(x); axR.set_xticklabels(bins, fontsize=9.5)
    axR.set_ylabel("paste 비율 (%)", fontsize=11)
    axR.set_title("(b) 스케일 bin 분포  (tail bin0+bin3 보강)", fontsize=12, fontweight="bold")
    axR.legend(fontsize=9.5)
    axR.grid(axis="y", ls=":", alpha=0.5)
    axR.spines[["top", "right"]].set_visible(False)
    save(fig, "slide05_inverse_freq.png")


# ============================================================
# Slide 6 — style match (concept diagram)
# ============================================================
def slide06():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), dpi=200)

    # (a) Lab histogram chi2 between host crop and candidate
    rng = np.random.default_rng(3)
    centers = np.linspace(0, 100, 32)
    host = np.exp(-((centers-55)**2)/420); host/=host.sum()
    good = np.exp(-((centers-58)**2)/470); good/=good.sum()
    bad  = np.exp(-((centers-30)**2)/520); bad/=bad.sum()
    axA = axes[0]
    axA.plot(centers, host, color="#333", lw=2.2, label="host crop")
    axA.fill_between(centers, good, color=C_D, alpha=0.45, label="후보 (가까움, 선택)")
    axA.fill_between(centers, bad, color=C_E, alpha=0.30, label="후보 (멀음, 제외)")
    axA.set_title("(a) Lab χ² 매칭\n비슷한 톤만 후보 (top-k=8)", fontsize=11, fontweight="bold")
    axA.set_xlabel("Lab 채널 값 (32-bin)", fontsize=9.5)
    axA.set_yticks([]); axA.legend(fontsize=8.5, loc="upper left")
    axA.spines[["top", "right", "left"]].set_visible(False)

    # (b) chi2 formula + selection bars
    axB = axes[1]; axB.axis("off")
    axB.text(0.5, 0.83, r"$\chi^2 = \sum_i \frac{(H_{host}(i)-H_{inst}(i))^2}{H_{host}(i)+H_{inst}(i)+\epsilon}$",
             ha="center", fontsize=14)
    cand = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    dist = [0.12, 0.18, 0.21, 0.27, 0.33, 0.41, 0.52, 0.66]
    wsel = (1/np.array(dist)); wsel/=wsel.sum()
    axB.set_title("(b) 거리 기반 가중샘플링", fontsize=11, fontweight="bold", y=0.62)
    inset = fig.add_axes([0.40, 0.12, 0.20, 0.32])
    inset.bar(cand, wsel, color=C_BLUE, edgecolor="white")
    inset.set_ylabel("선택 확률", fontsize=8)
    inset.tick_params(labelsize=7.5)
    inset.spines[["top", "right"]].set_visible(False)

    # (c) L-channel shift before/after
    axC = axes[2]
    cats = ["host\ncrop", "paste\n(before)", "paste\n(after)"]
    Lvals = [52, 71, 60]  # capped shift <=20 -> 71->60 (-11)
    cols = ["#333", C_E, C_D]
    bars = axC.bar(cats, Lvals, color=cols, width=0.6, edgecolor="white")
    axC.axhline(52, ls="--", color="#888", lw=1)
    axC.annotate("", xy=(2, 60), xytext=(1, 71),
                 arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=2))
    axC.text(1.5, 68, "ΔL 보정\n(|shift| ≤ 20)", ha="center", fontsize=9.5, color=C_BLUE)
    axC.set_ylabel("L-channel 평균", fontsize=10)
    axC.set_title("(c) 사후 명도(L) 보정\n고유 음영 보존", fontsize=11, fontweight="bold")
    axC.set_ylim(0, 90)
    axC.spines[["top", "right"]].set_visible(False)

    fig.suptitle("스타일 정합: 사전 Lab 매칭 + 사후 L-channel 보정 → paste artifact 억제",
                 fontsize=12.5, fontweight="bold", y=1.03)
    save(fig, "slide06_style_match.png")


# ============================================================
# Slide 7 — depth-free scale heuristic: flow + method share
# ============================================================
def slide07():
    fig = plt.figure(figsize=(12.5, 4.3), dpi=200)
    axL = fig.add_axes([0.04, 0.08, 0.52, 0.84]); axL.axis("off")
    axL.set_xlim(0, 10); axL.set_ylim(0, 10)

    steps = [
        ("① 동일 클래스 GT anchor", "같은 이미지의 동일 클래스 박스에서 스케일 차용", C_D, 7.6),
        ("② cy → h 선형 회귀", "클래스별 (세로위치 → 높이) 회귀 예측", C_BLUE, 4.9),
        ("③ scale bin 중심값", "샘플된 scale bin 중앙값으로 fallback", C_GT, 2.2),
    ]
    for title, desc, col, y in steps:
        box = FancyBboxPatch((0.4, y-0.85), 9.0, 1.55,
                             boxstyle="round,pad=0.12,rounding_size=0.18",
                             linewidth=2, edgecolor=col, facecolor=BOX_BG)
        axL.add_patch(box)
        axL.text(0.8, y+0.32, title, fontsize=12.5, fontweight="bold", color=col, va="center")
        axL.text(0.8, y-0.38, desc, fontsize=10, color="#444", va="center")
    for y0 in (6.75, 4.05):
        axL.annotate("", xy=(5, y0-0.55), xytext=(5, y0),
                     arrowprops=dict(arrowstyle="->", color="#888", lw=2))
        axL.text(5.5, y0-0.27, "실패 시", fontsize=9, color="#888", va="center")
    axL.text(5, 9.5, "Depth 추정 없이 데이터셋 통계 기반 3단계 fallback",
             ha="center", fontsize=12.5, fontweight="bold")

    # right: scale method share (real data) D vs E
    axR = fig.add_axes([0.66, 0.16, 0.31, 0.68])
    methods = ["gt_anchor\n①", "cy_reg.\n②", "target_scale\n③"]
    d = [28.9, 9.8, 61.4]
    e = [58.0, 13.3, 28.7]
    x = np.arange(3); w = 0.38
    axR.bar(x-w/2, d, w, label="Mode D", color=C_D)
    axR.bar(x+w/2, e, w, label="Mode E", color=C_E)
    axR.set_xticks(x); axR.set_xticklabels(methods, fontsize=9)
    axR.set_ylabel("사용 비율 (%)", fontsize=10)
    axR.set_title("스케일 결정 방식 분포", fontsize=11, fontweight="bold")
    axR.legend(fontsize=9)
    axR.grid(axis="y", ls=":", alpha=0.5)
    axR.spines[["top", "right"]].set_visible(False)
    axR.text(0.5, -0.30, "E: gt_anchor 58% — 맥락 기반 추정 비중↑",
             transform=axR.transAxes, ha="center", fontsize=9, color="#444")
    save(fig, "slide07_scale_heuristic.png")


# ============================================================
# Slide 8 — experiment / ablation matrix table
# ============================================================
def slide08():
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=200); ax.axis("off")
    rows = [
        ["A", "Real-only", "—", "—", "—", "—", "baseline"],
        ["B", "+ GT crop paste", "●", "—", "—", "—", "paste 자체 효과"],
        ["C", "+ SD pool random", "●", "●", "—", "—", "naive X-Paste"],
        ["D", "+ scene-aware", "●", "●", "●", "—", "scene-awareness"],
        ["E", "+ inverse-freq +\nstyle-match (Ours)", "●", "●", "●", "●", "제안 방법"],
    ]
    cols = ["Exp", "증강 구성", "Paste", "SD\npool", "Scene\naware", "Dist.+\nStyle", "Ablation 목적"]
    cw = [0.06, 0.26, 0.09, 0.08, 0.09, 0.10, 0.32]
    x0 = 0.02
    yt = 0.86; rh = 0.155
    # header
    x = x0
    for c, w in zip(cols, cw):
        ax.add_patch(plt.Rectangle((x, yt), w, rh, facecolor="#34495e", edgecolor="white"))
        ax.text(x+w/2, yt+rh/2, c, ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")
        x += w
    for i, r in enumerate(rows):
        y = yt - (i+1)*rh
        is_e = r[0] == "E"
        x = x0
        for j, (val, w) in enumerate(zip(r, cw)):
            bg = "#fdecef" if is_e else ("#f5f7fa" if i % 2 else "white")
            ax.add_patch(plt.Rectangle((x, y), w, rh, facecolor=bg, edgecolor="#dde3ea"))
            col = C_E if (val == "●") else ("#222")
            fw = "bold" if (is_e or j == 0 or val == "●") else "normal"
            fs = 10.5 if val == "●" else 9.8
            ax.text(x+w/2, y+rh/2, val, ha="center", va="center",
                    color=(C_E if is_e and j in (0,1) else col), fontsize=fs, fontweight=fw)
            x += w
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("실험 설계 — 점진적 Ablation (A→E)  |  YOLO11 n/s/m × 3 seeds",
                 fontsize=13, fontweight="bold", pad=10)
    ax.text(0.5, 0.02, "ID: Roboflow Military 1,934장 4-class   ·   OOD: 동일 도메인 별도 데이터셋",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#555")
    save(fig, "slide08_experiment_matrix.png")


# ============================================================
# Slide 9 — main results: ID & OOD grouped bars (yolo11m)
# ============================================================
def slide09():
    exps = ["A\nReal-only", "B\nGT paste", "C\nSD random", "D\nScene-aware", "E\nOurs"]
    id_map  = [90.2, 91.0, 91.2, 91.8, 92.5]
    ood_map = [34.7, 34.7, 35.5, 34.4, 38.0]
    colors = [C_GREY, C_GREY, C_GREY, C_GREY, C_E]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=200)
    for ax, vals, title, lo in [
        (axL, id_map,  "In-Distribution  mAP@0.5", 89),
        (axR, ood_map, "Out-of-Distribution  mAP@0.5", 33)]:
        bars = ax.bar(exps, vals, color=colors, width=0.62, edgecolor="white")
        for b, v, c in zip(bars, vals, colors):
            ax.text(b.get_x()+b.get_width()/2, v+ (max(vals)-lo)*0.02, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=11.5,
                    fontweight="bold", color=(C_E if c == C_E else "#333"))
        ax.set_ylim(lo, max(vals)+ (max(vals)-lo)*0.18)
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.tick_params(labelsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", ls=":", alpha=0.4)
    axL.set_ylabel("mAP@0.5", fontsize=11)
    # gain annotations
    axL.annotate(f"A→E +2.3", xy=(4, 92.5), xytext=(2.3, 92.7),
                 fontsize=10.5, color=C_E, fontweight="bold")
    axR.annotate(f"A→E +3.3", xy=(4, 38.0), xytext=(2.2, 37.6),
                 fontsize=10.5, color=C_E, fontweight="bold")
    fig.suptitle("YOLO11-m 정량 결과 — 제안(E)이 ID·OOD 모두 최고 (성능 저하 없이 일반화 향상)",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "slide09_results_main.png")


# ============================================================
# Slide 9b — full Table 1 rendered
# ============================================================
def slide09b():
    fig, ax = plt.subplots(figsize=(10.5, 3.4), dpi=200); ax.axis("off")
    header = ["", "ID mAP@.5", "ID mAP@.5:.95", "OOD mAP@.5", "OOD mAP@.5:.95"]
    data = [
        ["Real-only (A)",  "90.2", "74.4", "34.7", "22.1"],
        ["Copy-Paste (B)", "91.0", "75.3", "34.7", "22.7"],
        ["SD random (C)",  "91.2", "76.3", "35.5", "23.5"],
        ["Scene-aware (D)","91.8", "75.6", "34.4", "22.4"],
        ["Ours (E)",       "92.5", "76.8", "38.0", "24.0"],
    ]
    cw = [0.26, 0.185, 0.185, 0.185, 0.185]; x0 = 0.0
    yt = 0.82; rh = 0.15
    x = x0
    for c, w in zip(header, cw):
        ax.add_patch(plt.Rectangle((x, yt), w, rh, facecolor="#34495e", edgecolor="white"))
        ax.text(x+w/2, yt+rh/2, c, ha="center", va="center", color="white",
                fontsize=10.5, fontweight="bold")
        x += w
    best = [None, 92.5, 76.8, 38.0, 24.0]
    for i, r in enumerate(data):
        y = yt - (i+1)*rh
        is_e = "(E)" in r[0]
        x = x0
        for j, (val, w) in enumerate(zip(r, cw)):
            bg = "#fdecef" if is_e else ("#f5f7fa" if i % 2 else "white")
            ax.add_patch(plt.Rectangle((x, y), w, rh, facecolor=bg, edgecolor="#dde3ea"))
            isbest = (j > 0 and float(val) == best[j])
            ax.text(x+w/2, y+rh/2, val, ha="center", va="center",
                    color=(C_E if isbest else "#222"),
                    fontsize=10.8 if j else 10.2,
                    fontweight="bold" if (isbest or j == 0) else "normal")
            x += w
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(0, 1)
    ax.set_title("Table 1. YOLO11-m ID & OOD 성능 (제안 E 전부 최고)",
                 fontsize=13, fontweight="bold", pad=8)
    save(fig, "slide09b_results_table.png")


# ============================================================
# Slide 9c — model-size scaling (n/s/m) ID mAP50
# ============================================================
def slide09c():
    sizes = ["n", "s", "m"]
    data = {  # ID mAP50
        "A (Real-only)": [89.8, 90.1, 90.2],
        "D (Scene-aware)": [91.1, 91.5, 91.8],
        "E (Ours)": [91.9, 92.0, 92.5],
    }
    cmap = {"A (Real-only)": C_REAL, "D (Scene-aware)": C_D, "E (Ours)": C_E}
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    x = np.arange(3)
    for k, v in data.items():
        ax.plot(x, v, "-o", color=cmap[k], lw=2.4, markersize=8, label=k)
        for xi, yi in zip(x, v):
            ax.text(xi, yi+0.06, f"{yi:.1f}", ha="center", fontsize=9,
                    color=cmap[k], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"YOLO11-{s}" for s in sizes], fontsize=11)
    ax.set_ylabel("ID mAP@0.5", fontsize=11)
    ax.set_ylim(89, 93.2)
    ax.set_title("모델 크기 전반에서 일관된 우위 (E > D > A)", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(ls=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "slide09c_model_scaling.png")


if __name__ == "__main__":
    slide02(); slide03(); slide05(); slide06(); slide07()
    slide08(); slide09(); slide09b(); slide09c()
    print("ALL DONE ->", OUT)
