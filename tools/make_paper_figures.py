"""Generate Fig 2 (distribution comparison) and Fig 5 (ablation) for the paper.

Inputs are hardcoded from:
  - reports/aug_summary_v3.md  (per-class share, scale bin share)
  - reports/Final_Experimental_Results.md  (yolo11m mAP50 on ID/OOD-A/OOD-B)

Outputs:
  - reports/figures/fig2_distribution.png
  - reports/figures/fig5_ablation.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig2_distribution() -> Path:
    classes = ["Soldier", "civilian_veh.", "military_veh.", "persons"]
    real_gt = [26.0, 19.1, 16.4, 38.6]
    mode_d  = [19.5, 23.6, 20.7, 36.2]
    mode_e  = [28.8, 21.6, 22.5, 27.0]

    bins = ["bin0\n(small)", "bin1", "bin2", "bin3\n(large)"]
    scale_d = [82.5, 15.3, 2.2, 0.0]
    scale_e = [24.5, 28.4, 25.1, 22.1]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0), dpi=200)

    x = np.arange(len(classes))
    w = 0.27
    axL.bar(x - w, real_gt, w, label="Real GT",     color="#888888")
    axL.bar(x,     mode_d,  w, label="Mode D (uniform)", color="#4C9F70")
    axL.bar(x + w, mode_e,  w, label="Mode E (Ours)",    color="#D7263D")
    axL.set_xticks(x)
    axL.set_xticklabels(classes, fontsize=9)
    axL.set_ylabel("share of pastes (%)")
    axL.set_title("(a) Per-class paste share")
    axL.set_ylim(0, 45)
    axL.legend(fontsize=8, loc="upper right")
    axL.grid(axis="y", linestyle=":", alpha=0.5)

    x2 = np.arange(len(bins))
    w2 = 0.38
    axR.bar(x2 - w2/2, scale_d, w2, label="Mode D",       color="#4C9F70")
    axR.bar(x2 + w2/2, scale_e, w2, label="Mode E (Ours)", color="#D7263D")
    axR.set_xticks(x2)
    axR.set_xticklabels(bins, fontsize=9)
    axR.set_ylabel("share of pastes (%)")
    axR.set_title("(b) Scale-bin paste share")
    axR.set_ylim(0, 90)
    axR.legend(fontsize=8, loc="upper right")
    axR.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Fig. 2  Inverse-frequency sampler shifts paste distribution toward tail bins",
                 fontsize=10)
    fig.tight_layout()

    out = OUT_DIR / "fig2_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig5_ablation() -> Path:
    exps = ["A\nbaseline", "B\nReal Paste", "C\nNaive SD", "D\nScene-aware", "E\nOurs"]
    id_map50    = [0.902, 0.910, 0.912, 0.918, 0.925]
    ood_a_map50 = [0.133, 0.134, 0.214, 0.201, 0.182]
    ood_b_map50 = [0.347, 0.347, 0.355, 0.344, 0.380]

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=200)
    x = np.arange(len(exps))
    w = 0.27
    b1 = ax.bar(x - w, id_map50,    w, label="ID test",          color="#1F77B4")
    b2 = ax.bar(x,     ood_a_map50, w, label="OOD-A (military)", color="#FF7F0E")
    b3 = ax.bar(x + w, ood_b_map50, w, label="OOD-B (civilian)", color="#2CA02C")

    ax.set_xticks(x)
    ax.set_xticklabels(exps, fontsize=9)
    ax.set_ylabel("mAP@0.50")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fig. 5  YOLO11-m, 3-seed mean — E wins on ID and OOD-B; "
                 "C wins on OOD-A (trade-off)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out = OUT_DIR / "fig5_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p2 = fig2_distribution()
    p5 = fig5_ablation()
    print(f"saved: {p2}")
    print(f"saved: {p5}")
