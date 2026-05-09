"""Summarize aug provenance across paste modes B/C/D/E.

Reads `<aug_root>/aug_<MODE>/train/meta.jsonl` for each mode and produces
paper-ready marginal/joint distribution tables in markdown.

For inverse-frequency claim: compares per-class and per-scale-bin paste counts
across modes. Mode E (style_full) should oversample tail bins relative to
D (scene_uniform), since D is uniform and E is inverse-freq weighted.

Usage:
    python tools/summarize_aug_distribution.py \\
        --aug_root data/military_v1 \\
        --modes B C D E \\
        --hist cache/dist_v1.json \\
        --out reports/aug_summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

CLASSES = ["Soldier", "civilian_vehicle", "military_vehicle", "persons"]
N_SCALE_BINS = N_CX_BINS = N_CY_BINS = 4


def load_meta(meta_path: Path) -> list[dict]:
    if not meta_path.exists():
        return []
    out = []
    for line in meta_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def compute_stats(records: list[dict]) -> dict:
    n_imgs = len(records)
    n_accept = sum(1 for r in records if r.get("host_accept"))
    n_reject_by_reason: Counter = Counter()
    for r in records:
        if not r.get("host_accept"):
            n_reject_by_reason[r.get("reject_reason") or "unknown"] += 1

    pastes = [p for r in records for p in r.get("pastes", [])]
    total_pastes = len(pastes)

    per_class = Counter(p["cls"] for p in pastes)
    per_scale = Counter(p["sampled_bin"][1] for p in pastes if p.get("sampled_bin"))
    per_cx = Counter(p["sampled_bin"][2] for p in pastes if p.get("sampled_bin"))
    per_cy = Counter(p["sampled_bin"][3] for p in pastes if p.get("sampled_bin"))
    per_scale_method = Counter(p["scale_method"] for p in pastes)

    cy_scale = Counter()  # (cy_bin, scale_bin)
    for p in pastes:
        sb = p.get("sampled_bin")
        if sb:
            cy_scale[(sb[3], sb[1])] += 1

    src_counts: Counter = Counter()
    for p in pastes:
        src = p.get("instance_src")
        if src:
            src_counts[Path(src).name] += 1

    return {
        "n_imgs": n_imgs,
        "n_accept": n_accept,
        "accept_rate": n_accept / max(1, n_imgs),
        "n_reject_by_reason": dict(n_reject_by_reason),
        "total_pastes": total_pastes,
        "pastes_per_image": total_pastes / max(1, n_imgs),
        "pastes_per_accepted_image": total_pastes / max(1, n_accept),
        "per_class": dict(per_class),
        "per_scale_bin": dict(per_scale),
        "per_cx_bin": dict(per_cx),
        "per_cy_bin": dict(per_cy),
        "per_scale_method": dict(per_scale_method),
        "cy_scale_joint": {f"{cy},{s}": v for (cy, s), v in cy_scale.items()},
        "n_unique_pool_instances": len(src_counts),
        "top_pool_reuse": src_counts.most_common(10),
    }


def fmt_dist(counter: dict, keys: list, total: int) -> str:
    parts = []
    for k in keys:
        n = counter.get(k, 0)
        pct = (100.0 * n / total) if total else 0.0
        parts.append(f"{n} ({pct:.1f}%)")
    return " | ".join(parts)


def render_markdown(stats_per_mode: dict, hist_payload: dict | None, out_path: Path):
    lines: list[str] = []
    lines.append("# Aug provenance summary\n")
    if hist_payload is not None:
        per_class_count = hist_payload.get("per_class_count", {})
        n_total = hist_payload.get("n_instances_total", 0)
        lines.append("## Reference: train GT class distribution\n")
        lines.append("| class | GT count | share |")
        lines.append("|---|---:|---:|")
        for c in CLASSES:
            n = per_class_count.get(c, 0)
            pct = (100.0 * n / n_total) if n_total else 0.0
            lines.append(f"| {c} | {n} | {pct:.1f}% |")
        lines.append("")
        lines.append(f"_Total GT instances: {n_total}; train images: {hist_payload.get('n_train_imgs', 0)}_\n")

    # Top-level summary
    lines.append("## Summary across modes\n")
    lines.append("| mode | imgs | accept rate | total pastes | pastes / accepted img | unique pool inst |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode, s in stats_per_mode.items():
        lines.append(
            f"| {mode} | {s['n_imgs']} | {100*s['accept_rate']:.1f}% "
            f"| {s['total_pastes']} | {s['pastes_per_accepted_image']:.2f} "
            f"| {s['n_unique_pool_instances']} |"
        )
    lines.append("")

    # Per-class paste distribution
    lines.append("## Per-class paste counts (and share of total pastes within mode)\n")
    header = "| mode | " + " | ".join(CLASSES) + " | total |"
    sep = "|---|" + "---:|" * (len(CLASSES) + 1)
    lines.append(header)
    lines.append(sep)
    for mode, s in stats_per_mode.items():
        total = s["total_pastes"]
        cells = []
        for c in CLASSES:
            n = s["per_class"].get(c, 0)
            pct = (100.0 * n / total) if total else 0.0
            cells.append(f"{n} ({pct:.1f}%)")
        lines.append(f"| {mode} | " + " | ".join(cells) + f" | {total} |")
    lines.append("")
    lines.append("**Inverse-freq evidence**: in mode E, the share of underrepresented "
                 "classes (per the train GT histogram above) should be higher than in mode D "
                 "(uniform). If E's share for the rarest class exceeds D's by ≥5 pp, "
                 "the inverse-freq sampler is doing what we claim.\n")

    # Scale bin distribution
    lines.append("## Scale bin distribution (q0–q25 | q25–q50 | q50–q75 | q75+)\n")
    lines.append("| mode | bin 0 | bin 1 | bin 2 | bin 3 |")
    lines.append("|---|---:|---:|---:|---:|")
    for mode, s in stats_per_mode.items():
        total = sum(s["per_scale_bin"].values())
        lines.append(
            f"| {mode} | "
            + fmt_dist(s["per_scale_bin"], list(range(N_SCALE_BINS)), total)
            + " |"
        )
    lines.append("")
    lines.append("**Scale tail oversampling**: bins 0 and 3 are the tails. "
                 "If E's (bin 0 + bin 3) share exceeds D's, the sampler is biasing "
                 "toward small + large objects (paper claim).\n")

    # cx/cy distribution
    lines.append("## cx bin distribution (left → right)\n")
    lines.append("| mode | bin 0 | bin 1 | bin 2 | bin 3 |")
    lines.append("|---|---:|---:|---:|---:|")
    for mode, s in stats_per_mode.items():
        total = sum(s["per_cx_bin"].values())
        lines.append(
            f"| {mode} | "
            + fmt_dist(s["per_cx_bin"], list(range(N_CX_BINS)), total)
            + " |"
        )
    lines.append("")

    lines.append("## cy bin distribution (top → bottom)\n")
    lines.append("| mode | bin 0 | bin 1 | bin 2 | bin 3 |")
    lines.append("|---|---:|---:|---:|---:|")
    for mode, s in stats_per_mode.items():
        total = sum(s["per_cy_bin"].values())
        lines.append(
            f"| {mode} | "
            + fmt_dist(s["per_cy_bin"], list(range(N_CY_BINS)), total)
            + " |"
        )
    lines.append("")

    # Scale method (E only meaningful)
    lines.append("## Scale method distribution (depth-free fallback chain)\n")
    methods = sorted({m for s in stats_per_mode.values() for m in s["per_scale_method"]})
    if methods:
        header = "| mode | " + " | ".join(methods) + " | total |"
        sep = "|---|" + "---:|" * (len(methods) + 1)
        lines.append(header)
        lines.append(sep)
        for mode, s in stats_per_mode.items():
            total = sum(s["per_scale_method"].values())
            cells = []
            for m in methods:
                n = s["per_scale_method"].get(m, 0)
                pct = (100.0 * n / total) if total else 0.0
                cells.append(f"{n} ({pct:.1f}%)")
            lines.append(f"| {mode} | " + " | ".join(cells) + f" | {total} |")
    else:
        lines.append("(no scale method data)")
    lines.append("")
    lines.append("`gt_anchor` = scaled from a same-class GT box on this image; "
                 "`cy_regression` = predicted from per-class cy → h fit; "
                 "`target_scale` = fallback to sampled scale bin midpoint.\n")

    # Reject reasons
    lines.append("## Host accept/reject\n")
    all_reasons = sorted({r for s in stats_per_mode.values() for r in s["n_reject_by_reason"]})
    if all_reasons:
        lines.append("| mode | accepted | " + " | ".join(all_reasons) + " |")
        lines.append("|---|---:|" + "---:|" * len(all_reasons))
        for mode, s in stats_per_mode.items():
            cells = [str(s["n_accept"])] + [str(s["n_reject_by_reason"].get(r, 0)) for r in all_reasons]
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    else:
        lines.append("(all images accepted in all modes)")
    lines.append("")

    # Top reused pool instances (E mainly)
    lines.append("## Top-10 reused SD pool instances per mode\n")
    for mode, s in stats_per_mode.items():
        lines.append(f"### {mode}")
        if not s["top_pool_reuse"]:
            lines.append("(no SD pool provenance — likely real_random mode)")
            lines.append("")
            continue
        lines.append("| rank | filename | reuse count |")
        lines.append("|---:|---|---:|")
        for i, (name, n) in enumerate(s["top_pool_reuse"], 1):
            lines.append(f"| {i} | `{name}` | {n} |")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"-> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aug_root", required=True,
                    help="root containing aug_<MODE>/train/meta.jsonl (e.g. data/military_v1)")
    ap.add_argument("--modes", nargs="+", default=["B", "C", "D", "E"])
    ap.add_argument("--hist", default=None, help="optional cache/dist_v1.json for GT reference table")
    ap.add_argument("--out", default="reports/aug_summary.md")
    args = ap.parse_args()

    aug_root = Path(args.aug_root)
    stats_per_mode: dict[str, dict] = {}
    for mode in args.modes:
        meta_path = aug_root / f"aug_{mode}" / "train" / "meta.jsonl"
        records = load_meta(meta_path)
        if not records:
            print(f"[skip] {meta_path} missing or empty", file=sys.stderr)
            continue
        stats_per_mode[mode] = compute_stats(records)
        print(f"[load] mode {mode}: {len(records)} imgs, {stats_per_mode[mode]['total_pastes']} pastes")

    if not stats_per_mode:
        print("no meta.jsonl found in any mode", file=sys.stderr)
        return 1

    hist_payload = None
    if args.hist:
        try:
            hist_payload = json.loads(Path(args.hist).read_text())
        except FileNotFoundError:
            print(f"[skip] hist {args.hist} not found", file=sys.stderr)

    render_markdown(stats_per_mode, hist_payload, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
