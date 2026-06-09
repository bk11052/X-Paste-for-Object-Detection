# -*- coding: utf-8 -*-
"""Redraw OUR distribution-aware copy-paste pipeline in the clean left-to-right
style of the reference architecture figure (pastel rounded boxes, colored
borders, bottom phase brackets, non-overlapping orthogonal arrows).

Order follows xpaste/aug/build_augmented.py (style-match consumes the host crop
at the planned bbox, so it sits AFTER placement)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle

for cand in ["Apple SD Gothic Neo", "AppleGothic", "Nanum Gothic", "DejaVu Sans"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("/Users/kyu216/Univ/학부연구생/논문/02. MilitaryDataset/architecture")
OUT.mkdir(parents=True, exist_ok=True)

BLUE = ("#dbe6f5", "#3f6fb0")   # data / modeling
GRN  = ("#dcecd9", "#5a9a52")   # host / segformer
ORG  = ("#fbe7cf", "#d98a2b")   # distribution-aware decision (what/where)
PINK = ("#f7dee4", "#c25b73")   # style-matched synthesis
GATE = ("#fdf3da", "#cda732")   # accept gate
OUTC = ("#d8f0d8", "#2e8b3d")   # output
boxes = {}


def box(ax, bid, cx, cy, w, h, text, colors, fs=9.5, bold=True, rnd=0.5):
    fill, edge = colors
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle=f"round,pad=0.12,rounding_size={rnd}",
                 linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=4))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#1b2733", zorder=5)
    boxes[bid] = (cx, cy, w, h)


def ortho(ax, pts, color="#33414f", lw=2.0, ls="-"):
    for i in range(len(pts)-2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=color, lw=lw, ls=ls, zorder=3, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                 mutation_scale=14, lw=lw, color=color, ls=ls, zorder=3))


def lab(ax, x, y, t, fs=8.4, color="#3a4654", weight="normal"):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, color=color,
            fontweight=weight, zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92))


def thumb(ax, cx, cy, w, h, kind="img"):
    """tiny decorative thumbnail to evoke the reference feel."""
    ax.add_patch(Rectangle((cx-w/2, cy-h/2), w, h, facecolor="#eef1f4",
                 edgecolor="#9aa4af", lw=1.2, zorder=5))
    if kind == "img":
        ax.add_patch(Polygon([(cx-w/2+1, cy-h/2+1), (cx-1, cy+1), (cx+w/2-1, cy-h/2+1)],
                     closed=True, facecolor="#bcc8d6", edgecolor="none", zorder=6))
        ax.add_patch(Circle((cx+w/2-2.4, cy+h/2-2.2), 0.9, facecolor="#e9c46a",
                     edgecolor="none", zorder=6))
    elif kind == "hist":
        for i, bh in enumerate([2.0, 4.2, 3.0, 5.0]):
            ax.add_patch(Rectangle((cx-w/2+1.2+i*2.0, cy-h/2+1), 1.4, bh,
                         facecolor="#7fa6d6", edgecolor="none", zorder=6))
    elif kind == "pool":
        cols = ["#8fae8a", "#c79a9f", "#9db8e0", "#d3b070"]
        for i, c in enumerate(cols):
            r, cc = divmod(i, 2)
            ax.add_patch(Rectangle((cx-w/2+1.6+cc*4.2, cy-h/2+1.4+r*3.6), 3.2, 2.6,
                         facecolor=c, edgecolor="none", zorder=6))


def main():
    fig, ax = plt.subplots(figsize=(21, 9), dpi=200)
    ax.set_xlim(0, 236); ax.set_ylim(2, 100); ax.axis("off")
    ax.add_patch(FancyBboxPatch((3, 9), 230, 86, boxstyle="round,pad=0.2,rounding_size=2",
                 linewidth=0, facecolor="#fafbfc", zorder=0))

    # per-accepted-image loop backdrop
    ax.add_patch(FancyBboxPatch((102, 40), 100, 40, boxstyle="round,pad=0.2,rounding_size=2",
                 linewidth=1.5, edgecolor="#d98a2b", facecolor="#fdf8f1",
                 linestyle=(0, (5, 3)), zorder=1))
    ax.text(104, 78, "accept된 host 이미지마다  ·  paste 시도 × N 반복",
            fontsize=9, color="#b9772a", fontweight="bold", zorder=2)

    # ---------- input column ----------
    box(ax, "TrainGT", 15, 82, 20, 9, "Train GT\nLabels", BLUE, fs=9.5)
    box(ax, "HostImg", 15, 50, 22, 13, "", GRN)
    thumb(ax, 15, 52, 16, 8, "img"); ax.text(15, 44.5, "Host Train Image", ha="center",
          fontsize=9, fontweight="bold", zorder=6)
    box(ax, "SDPool", 15, 18, 20, 11, "", PINK)
    thumb(ax, 15, 20, 15, 7, "pool"); ax.text(15, 13.5, "SD 1.5 Pool", ha="center",
          fontsize=9, fontweight="bold", zorder=6)

    # ---------- modeling column ----------
    box(ax, "Hist", 48, 82, 24, 12, "4D Joint Histogram\n(class×scale×cx×cy)\n+ cy→h 회귀", BLUE, fs=8.6)
    box(ax, "SegF", 48, 50, 24, 12, "SegFormer\nScene Parsing\n(ground / road)", GRN, fs=8.8)
    box(ax, "PoolIdx", 48, 18, 24, 10, "Pool Index\n(클래스별 Lab hist)", PINK, fs=8.6)

    # ---------- sampling / accept ----------
    box(ax, "Sampler", 86, 82, 26, 13, "Inverse-Freq Sampler\n(T=2.0)\n→ target (c, s, cx, cy)", ORG, fs=8.6)
    box(ax, "Accept", 86, 50, 22, 12, "Host\nAccept / Reject\n게이트", GATE, fs=8.6)
    box(ax, "Reject", 86, 30, 28, 9, "Reject: paste 없이\n원본 그대로 복사", ("#eef1f4", "#9aa4af"), fs=8.2)

    # ---------- synthesis spine ----------
    box(ax, "Scale", 126, 69, 26, 12, "Depth-free\nScale Heuristic\n→ h, w (px)", ORG, fs=8.6)
    box(ax, "Place", 126, 50, 26, 12, "Placement Planner\nregion + IoU\n→ bbox", ORG, fs=8.6)
    box(ax, "Style", 162, 50, 26, 13, "Lab χ² Style-match\nhost crop ↔ pool\n→ instance", PINK, fs=8.5)
    box(ax, "Compose", 196, 50, 22, 12, "Alpha-blend\n+ Post L-match", PINK, fs=8.8)

    # ---------- output container ----------
    ax.add_patch(FancyBboxPatch((212, 22), 22, 56, boxstyle="round,pad=0.2,rounding_size=1.5",
                 linewidth=1.8, edgecolor="#2e8b3d", facecolor="#f0f8ef", zorder=2))
    ax.text(223, 84, "Augmented\nDataset", ha="center", va="center",
            fontsize=10.5, fontweight="bold", zorder=5)
    box(ax, "OImg", 223, 66, 18, 11, "Augmented\nImages", OUTC, fs=8.8)
    box(ax, "OLbl", 223, 50, 18, 11, "+ Merged\nYOLO Labels", OUTC, fs=8.8)
    box(ax, "OTrain", 223, 34, 18, 11, "→ Train\nYOLO11 n/s/m", OUTC, fs=8.8)

    # ---------- arrows ----------
    ortho(ax, [(25, 82), (35.5, 82)])            # TrainGT -> Hist
    ortho(ax, [(26, 50), (35.5, 50)])            # HostImg -> SegF
    ortho(ax, [(25, 18), (35.5, 18)])            # SDPool -> PoolIdx
    ortho(ax, [(60, 82), (72.5, 82)])            # Hist -> Sampler
    ortho(ax, [(60, 50), (74.5, 50)])            # SegF -> Accept

    # Sampler -> Scale (target conditions)
    ortho(ax, [(90, 75.5), (90, 69), (112.5, 69)], color="#d98a2b")
    lab(ax, 100, 73.5, "target (c, s, cx, cy)", color="#b9772a", fs=8.2)
    # Scale -> Place
    ortho(ax, [(126, 63), (126, 56)], color="#d98a2b")
    # Accept -> Place  (ACCEPT branch: accepted host + region mask -> synthesis loop)
    ortho(ax, [(97, 50), (112.5, 50)], color="#5a9a52")
    lab(ax, 105, 53.6, "accept · region mask", color="#3f7a44", fs=7.6)
    # REJECT branch: gate -> reject box -> straight to output (image copied, no paste)
    ortho(ax, [(86, 44), (86, 34.5)], color="#b0392a")
    lab(ax, 93.8, 39.6, "reject", color="#b0392a", fs=8.0, weight="bold")
    ortho(ax, [(72, 30), (55, 30), (55, 12), (218, 12), (218, 21.5)],
          color="#9aa4af", ls=(0, (5, 3)))
    lab(ax, 132, 12, "rejected: 원본 그대로 (paste 0)", color="#6b7682", fs=7.8)
    # Place -> Style (host crop)
    ortho(ax, [(139, 50), (148.5, 50)])
    lab(ax, 143.7, 53.2, "host crop", fs=8.0)
    # PoolIdx -> Style (pool candidates, bottom routing)
    ortho(ax, [(60, 18), (162, 18), (162, 43.2)], color="#c25b73")
    lab(ax, 110, 14.6, "pool candidates", color="#a8506a", fs=8.4)
    # Style -> Compose
    ortho(ax, [(175, 50), (184.8, 50)])
    # Compose -> output container
    ortho(ax, [(207, 50), (213.5, 50)])
    # container internal fan (image/label -> train)
    ortho(ax, [(223, 60.5), (223, 56)], color="#2e8b3d", lw=1.6)
    ortho(ax, [(223, 44.5), (223, 40)], color="#2e8b3d", lw=1.6)

    # ---------- phase brackets ----------
    phases = [(6, 70, "Scene & Distribution\nModeling"),
              (72, 146, "Distribution-aware\nSampling & Placement"),
              (148, 208, "Style-matched\nSynthesis"),
              (210, 234, "Augmented\nDataset")]
    yb = 7.0
    for x0, x1, name in phases:
        ax.plot([x0, x1], [yb, yb], color="#6b7682", lw=1.4, zorder=1)
        for xx in (x0, x1):
            ax.plot([xx, xx], [yb-0.9, yb+0.9], color="#6b7682", lw=1.4, zorder=1)
        ax.text((x0+x1)/2, yb-2.6, name, ha="center", va="top", fontsize=9.5,
                color="#46545f", fontweight="bold")

    fig.savefig(OUT / "my_pipeline_styled.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "my_pipeline_styled.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved ->", OUT / "my_pipeline_styled.png")


if __name__ == "__main__":
    main()
