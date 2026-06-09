# -*- coding: utf-8 -*-
"""Redraw the corrected pipeline figure (Fig. 1) reflecting the true execution
order in xpaste/aug/build_augmented.py.

Key correction vs the old figure: the Lab-χ² style-match consumes the HOST CROP
at the already-planned bbox (style_match.py:select_topk_match), so it sits AFTER
placement — not as an upstream parallel lane. Also adds the depth-free scale
heuristic and the placement planner as explicit steps, and shows SegFormer's two
roles (accept gate + region mask for placement).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

for cand in ["Apple SD Gothic Neo", "AppleGothic", "Nanum Gothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("/Users/kyu216/Univ/학부연구생/논문/02. MilitaryDataset/pipeine")
OUT.mkdir(parents=True, exist_ok=True)

# palette: (fill, edge)
OFF = ("#dbe7f3", "#2F6DB5")   # offline / data
SEG = ("#dcefe2", "#3f9f6e")   # segformer / host
GATE = ("#fff3bf", "#d4a017")  # accept gate
DEC = ("#fde8d6", "#e0892a")   # decide: sampler / scale / placement (what & where)
STY = ("#fbe0e4", "#D7263D")   # style / blend (how natural)
OUTC = ("#d8f0d8", "#2e8b3d")  # output
PLAIN = ("#eef1f4", "#9aa6b2")

boxes = {}  # id -> (cx, cy, w, h)


def box(ax, bid, cx, cy, w, h, text, colors, fs=9, bold=False, rounding=0.6):
    fill, edge = colors
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle=f"round,pad=0.15,rounding_size={rounding}",
                       linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1b2733", zorder=4)
    boxes[bid] = (cx, cy, w, h)


def side(bid, s):
    cx, cy, w, h = boxes[bid]
    return {
        "top": (cx, cy + h / 2), "bottom": (cx, cy - h / 2),
        "left": (cx - w / 2, cy), "right": (cx + w / 2, cy),
    }[s]


def arrow(ax, p1, p2, color="#33414f", lw=2.0, ls="-", style="-|>",
          rad=0.0, zorder=5):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                        linewidth=lw, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=zorder)
    ax.add_patch(a)


def connect(ax, a, sa, b, sb, **kw):
    arrow(ax, side(a, sa), side(b, sb), **kw)


def band(ax, x0, y0, x1, y1, label, face="#f4f7fa", edge="#c4cdd6", ls="-"):
    r = Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=face, edgecolor=edge,
                  linewidth=1.6, linestyle=ls, zorder=1)
    ax.add_patch(r)
    ax.text(x0 + 1.5, y1 - 2.2, label, ha="left", va="top", fontsize=10.5,
            fontweight="bold", color="#46566a", zorder=2)


def main():
    fig, ax = plt.subplots(figsize=(15, 13), dpi=200)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 132)
    ax.axis("off")

    # ---------------- OFFLINE band ----------------
    band(ax, 4, 107, 146, 131, "오프라인 사전계산 (1회)")
    box(ax, "A1", 22, 122, 24, 9, "Train GT\nlabels", OFF)
    box(ax, "A2", 78, 122, 50, 13,
        "4D Joint Histogram\n(class × scale × cx × cy)\n+ scale quartiles · cy→h 회귀", OFF, fs=9)
    box(ax, "B1", 22, 111, 24, 8, "SD 1.5\nInstance Pool", OFF)
    box(ax, "B2", 78, 111, 50, 8, "Pool Index  (클래스별 Lab 히스토그램)", OFF)
    connect(ax, "A1", "right", "A2", "left")
    connect(ax, "B1", "right", "B2", "left")

    # ---------------- PER-IMAGE band ----------------
    band(ax, 4, 3, 146, 104,
         "host 이미지 1장마다  —  accept ⇒ paste 시도 ×N 반복", face="#fbfcfd",
         edge="#7f8c9b", ls=(0, (6, 4)))

    # Step 1: host + segformer + accept gate
    box(ax, "S1", 18, 95, 22, 9, "Host real\nimage", SEG)
    box(ax, "S2", 58, 95, 38, 12,
        "SegFormer ADE20K 파싱\n→ region mask (ground/road)\n+ GT boxes", SEG, fs=9)
    box(ax, "S3", 98, 95, 26, 11, "Accept?\n(paste-region /\nGT density)", GATE, fs=9, bold=True)
    box(ax, "RJ", 130, 95, 20, 8, "원본 그대로\n복사", PLAIN, fs=8.5)
    connect(ax, "S1", "right", "S2", "left")
    connect(ax, "S2", "right", "S3", "left")
    connect(ax, "S3", "right", "RJ", "left")
    ax.text(114, 98.5, "reject", ha="center", fontsize=8.5, color="#b04a2a", fontweight="bold")

    # ---------------- inner per-paste box ----------------
    band(ax, 8, 27, 142, 85, "paste 시도 1회마다", face="#fdf6ef",
         edge="#e0a060", ls=(0, (5, 3)))

    # Row 1 (a -> b -> c)
    box(ax, "a", 32, 73, 36, 12,
        "Inverse-Freq Sampler\n→ target (c, scale, cx, cy)", DEC, fs=9, bold=True)
    box(ax, "b", 76, 73, 36, 12,
        "Depth-free Scale Heuristic\ngt_anchor → cy-회귀 → bin중심\n→ h, w (px)", DEC, fs=8.7)
    box(ax, "c", 120, 73, 36, 12,
        "Placement Planner\nregion + frame + IoU reject\n→ bbox", DEC, fs=8.7)
    connect(ax, "a", "right", "b", "left")
    connect(ax, "b", "right", "c", "left")

    # Row 2 (c -> d -> e -> f), right to left
    box(ax, "d", 120, 53, 30, 9, "Host Crop\n@ bbox", STY, fs=9)
    box(ax, "e", 76, 53, 38, 13,
        "Lab χ² Style-match\nhost crop ↔ pool[c], top-k=8\n→ selected instance", STY, fs=8.7, bold=True)
    box(ax, "f", 32, 53, 30, 9, "Alpha-blend\ninstance → host", STY, fs=9)
    connect(ax, "c", "bottom", "d", "top")
    connect(ax, "d", "left", "e", "right")
    connect(ax, "e", "left", "f", "right")

    # f -> down -> g
    box(ax, "g", 32, 35, 36, 9, "Post L-channel match\n(|Δ| ≤ 20)", STY, fs=9)
    connect(ax, "f", "bottom", "g", "top")

    # loop-back note g -> a
    arrow(ax, (14, 35), (14, 73), color="#c08a3a", lw=1.4, ls=(0, (4, 3)), rad=0.0)
    arrow(ax, side("g", "left"), (14, 35), color="#c08a3a", lw=1.4, ls=(0, (4, 3)))
    arrow(ax, (14, 73), side("a", "left"), color="#c08a3a", lw=1.4, ls=(0, (4, 3)))
    ax.text(11.2, 54, "다음 시도 반복", ha="center", va="center", fontsize=8,
            color="#b07a2a", rotation=90)

    # ---------------- Step 3: merge + output ----------------
    box(ax, "M", 50, 15, 42, 9, "GT + paste box 병합\n→ YOLO label", OFF, fs=9)
    box(ax, "OUT", 112, 15, 40, 12, "Augmented YOLO\nTraining Set", OUTC, fs=10.5, bold=True)
    connect(ax, "M", "right", "OUT", "left")
    # g exits inner box -> M
    arrow(ax, (32, 30.5), (40, 19.5), color="#33414f", lw=2.0)
    ax.text(30, 24, "N회 종료 후", ha="left", fontsize=8.2, color="#5a6675")
    # reject path -> OUT
    arrow(ax, side("RJ", "bottom"), side("OUT", "top"), color="#8a96a4",
          lw=1.6, ls=(0, (4, 3)), rad=-0.25)

    # ---------------- cross dependencies (dashed, colored) ----------------
    # joint hist -> sampler (a)
    arrow(ax, (60, side("A2", "bottom")[1]), side("a", "top"),
          color="#e0892a", lw=1.6, ls=(0, (5, 3)), rad=0.15)
    ax.text(40, 88, "joint hist", fontsize=8, color="#c2761f", fontweight="bold")
    # cy regression -> scale heuristic (b)
    arrow(ax, (90, side("A2", "bottom")[1]), side("b", "top"),
          color="#e0892a", lw=1.3, ls=(0, (3, 3)), rad=-0.12)
    ax.text(92.5, 90, "cy→h fit", fontsize=7.6, color="#c2761f")
    # pool index -> style-match (e)
    arrow(ax, (96, side("B2", "bottom")[1]), side("e", "right"),
          color="#D7263D", lw=1.6, ls=(0, (5, 3)), rad=-0.35)
    ax.text(99, 78, "pool", fontsize=8, color="#c01f34", fontweight="bold")
    # region mask -> placement (c)
    arrow(ax, side("S2", "bottom"), (side("c", "top")[0] - 4, side("c", "top")[1]),
          color="#2e8b3d", lw=1.6, ls=(0, (5, 3)), rad=-0.25)
    ax.text(96, 87.5, "region mask", fontsize=8, color="#247033", fontweight="bold")

    # ---------------- legend ----------------
    leg = [("데이터 / 인덱스", OFF), ("host · SegFormer", SEG),
           ("무엇·어디 결정", DEC), ("자연스러운 합성", STY),
           ("accept 게이트", GATE), ("출력", OUTC)]
    lx = 9
    for name, (fill, edge) in leg:
        ax.add_patch(FancyBboxPatch((lx, 0.2), 3.2, 2.4,
                     boxstyle="round,pad=0.1,rounding_size=0.4",
                     linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=6))
        ax.text(lx + 4, 1.4, name, fontsize=8.3, va="center", zorder=6)
        lx += 23

    ax.set_title("Fig. 1  Distribution-aware Generative Copy-Paste — 실제 실행 순서",
                 fontsize=15, fontweight="bold", pad=14)

    p = OUT / "pipeline_fig1_corrected.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "pipeline_fig1_corrected.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)


if __name__ == "__main__":
    main()
