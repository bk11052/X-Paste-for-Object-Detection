# -*- coding: utf-8 -*-
"""Clean left-to-right redraw of the vectorized planning architecture with
non-overlapping orthogonal arrows and dedicated routing channels."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

for cand in ["Apple SD Gothic Neo", "AppleGothic", "Nanum Gothic", "DejaVu Sans"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("/Users/kyu216/Univ/학부연구생/논문/02. MilitaryDataset/architecture")
OUT.mkdir(parents=True, exist_ok=True)

# (fill, edge)
GRAY = ("#f1f3f6", "#9aa4af")
BLUE = ("#dbe6f5", "#3f6fb0")
PINK = ("#f7dee4", "#c25b73")
VEC  = ("#eef1f4", "#9aa4af")
GRN  = ("#dcecd9", "#5a9a52")
CON  = ("#e6f0e3", "#6aa05f")
GATE = ("#fdf3da", "#cda732")
TBLUE, TGRN, TRED = "#9db8e0", "#94c79a", "#e3a0a8"

boxes = {}


def box(ax, bid, cx, cy, w, h, text, colors, fs=10.5, bold=True, rnd=0.5):
    fill, edge = colors
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle=f"round,pad=0.12,rounding_size={rnd}",
                 linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1b2733", zorder=4)
    boxes[bid] = (cx, cy, w, h)


def ortho(ax, pts, color="#33414f", lw=2.0, ls="-"):
    for i in range(len(pts)-2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                 mutation_scale=14, lw=lw, color=color, ls=ls, zorder=2))


def lab(ax, x, y, t, fs=8.6, color="#3a4654", rot=0, weight="normal"):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, color=color,
            rotation=rot, fontweight=weight, zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92))


def tokens(ax, cx, cy):
    """three stacked ego-query tokens"""
    for dy, c in [(4.6, TBLUE), (0, TGRN), (-4.6, TRED)]:
        ax.add_patch(FancyBboxPatch((cx-4, cy+dy-1.8), 8, 3.6,
                     boxstyle="round,pad=0.05,rounding_size=0.3",
                     linewidth=1.2, edgecolor="#5a6675", facecolor=c, zorder=4))


def main():
    fig, ax = plt.subplots(figsize=(20, 9.2), dpi=200)
    ax.set_xlim(0, 218); ax.set_ylim(2, 100); ax.axis("off")
    # outer panel
    ax.add_patch(FancyBboxPatch((3, 9), 211, 86, boxstyle="round,pad=0.2,rounding_size=2",
                 linewidth=0, facecolor="#fafbfc", zorder=0))

    # ---- boxes ----
    box(ax, "MVImg", 13, 50, 16, 13, "Multi-view\nImages", GRAY, fs=10)
    box(ax, "BEVEnc", 35, 50, 15, 11, "BEV\nEncoder", GRAY)
    box(ax, "BEVFeat", 55, 50, 13, 11, "BEV\nFeatures", ("#f6ead0", "#c9a843"), fs=9.5)
    box(ax, "AgentQ", 13, 82, 18, 8, "Agent Query", BLUE, fs=10)
    box(ax, "MapQ", 13, 18, 18, 8, "Map Query", PINK, fs=10)

    box(ax, "MotionTf", 85, 82, 27, 13, "Vectorized Motion\nTransformer", BLUE)
    box(ax, "MapTf", 85, 18, 27, 13, "Vectorized Map\nTransformer", PINK)
    box(ax, "MotionVec", 115, 82, 15, 12, "Motion\nVector", VEC, fs=10)
    box(ax, "MapVec", 115, 18, 15, 12, "Map\nVector", VEC, fs=10)

    box(ax, "EgoQ", 124, 50, 13, 19, "", VEC, fs=10)
    ax.text(124, 61.5, "Ego Query", ha="center", fontsize=9.5, fontweight="bold", zorder=5)
    tokens(ax, 124, 49)

    box(ax, "EgoStatus", 152, 72, 19, 8, "Ego Status *", GATE, fs=10)
    box(ax, "PlanTf", 152, 50, 23, 13, "Planning\nTransformer", GRN)
    box(ax, "DriveCmd", 152, 28, 21, 9, '“Turn Left”\nDriving Command', GATE, fs=9.5)
    box(ax, "EgoVec", 178, 50, 15, 12, "Ego\nVector", VEC, fs=10)

    # constraints container + 3
    ax.add_patch(FancyBboxPatch((186, 18), 28, 64, boxstyle="round,pad=0.2,rounding_size=1.5",
                 linewidth=1.8, edgecolor="#6aa05f", facecolor="#f1f7ef", zorder=2))
    ax.text(200, 88, "Vectorized\nPlanning Constraints", ha="center", va="center",
            fontsize=11.5, fontweight="bold", zorder=5)
    box(ax, "Coll", 200, 71, 23, 12, "Ego-Agent\nCollision\nConstraint", CON, fs=9.8)
    box(ax, "Bound", 200, 50, 23, 12, "Ego-Boundary\nOverstepping\nConstraint", CON, fs=9.8)
    box(ax, "Lane", 200, 29, 23, 12, "Ego-Lane\nDirectional\nConstraint", CON, fs=9.8)

    # ---- arrows (clean orthogonal, dedicated channels) ----
    ortho(ax, [(21, 50), (27.5, 50)])
    ortho(ax, [(42.5, 50), (48.5, 50)])
    ortho(ax, [(22, 82), (71.5, 82)])            # AgentQ -> MotionTf
    ortho(ax, [(22, 18), (71.5, 18)])            # MapQ  -> MapTf
    ortho(ax, [(55, 55.5), (55, 78), (71.5, 78)])  # BEVFeat -> MotionTf
    ortho(ax, [(55, 44.5), (55, 22), (71.5, 22)])  # BEVFeat -> MapTf
    ortho(ax, [(78, 24.5), (78, 75.5)], color="#c25b73")  # Updated Map Query -> Motion
    lab(ax, 72.5, 50, "Updated\nMap Query", color="#a8506a")

    ortho(ax, [(98.5, 82), (107.5, 82)])         # MotionTf -> MotionVec
    ortho(ax, [(98.5, 18), (107.5, 18)])         # MapTf -> MapVec

    # context to Ego Query (separate y-ranges on channel x=112)
    ortho(ax, [(112, 76), (112, 55), (117.5, 55)], color="#3f6fb0")
    lab(ax, 106, 67, "Updated\nAgent Query", color="#37608f")
    ortho(ax, [(112, 24), (112, 45), (117.5, 45)], color="#c25b73")
    lab(ax, 106, 33, "Map\ncontext", color="#a8506a")

    # Ego Query -> Planning (k,v / q / k,v)
    ortho(ax, [(130.5, 54), (140.5, 54)])
    ortho(ax, [(130.5, 50), (140.5, 50)])
    ortho(ax, [(130.5, 46), (140.5, 46)])
    lab(ax, 135.5, 56.4, "k, v", fs=8.2); lab(ax, 135.5, 50, "q", fs=8.2)
    lab(ax, 135.5, 43.6, "k, v", fs=8.2)

    ortho(ax, [(152, 68), (152, 56.5)])          # Ego Status -> Plan
    ortho(ax, [(152, 32.5), (152, 43.5)])        # Driving cmd -> Plan
    ortho(ax, [(163.5, 50), (170.5, 50)])        # Plan -> EgoVec

    # outputs -> constraints
    ortho(ax, [(122.5, 82), (193, 82), (193, 77.2)])   # Motion Vector -> Collision
    lab(ax, 150, 85.5, "Motion Vector", fs=9, weight="bold")
    ortho(ax, [(185.5, 50), (188.5, 50)])              # Ego Vector -> Boundary
    lab(ax, 178, 43.5, "Ego Vector", fs=9, weight="bold")
    ortho(ax, [(122.5, 18), (193, 18), (193, 22.8)])   # Map Vector -> Lane
    lab(ax, 150, 14.5, "Map Vector", fs=9, weight="bold")

    # ---- phase brackets ----
    phases = [(5, 65, "Backbone"), (67, 131, "Vectorized\nScene Learning"),
              (133, 169, "Planning\nInferring Phase"), (171, 213, "Planning\nTraining Phase")]
    yb = 7.0
    for x0, x1, name in phases:
        ax.plot([x0, x1], [yb, yb], color="#6b7682", lw=1.4, zorder=1)
        for xx in (x0, x1):
            ax.plot([xx, xx], [yb-0.9, yb+0.9], color="#6b7682", lw=1.4, zorder=1)
        ax.text((x0+x1)/2, yb-3.2, name, ha="center", va="top", fontsize=10,
                color="#46545f", fontweight="bold")

    fig.savefig(OUT / "architecture_redraw.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "architecture_redraw.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved ->", OUT / "architecture_redraw.png")


if __name__ == "__main__":
    main()
