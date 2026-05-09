"""Filter SD instance pool by CLIP margin between intended class and a confusing class.

Used to mitigate Soldier vs persons label leakage in the new 4-class dataset:
  - For each Soldier instance, require CLIP("soldier in combat uniform with rifle") -
    CLIP("civilian person in casual clothes") >= margin.
  - For each persons instance, require the inverse margin >= margin.
Ambiguous instances are dropped (not copied to output).

Pool layout (from segment_pose_hf.py): pool_dir/<category>__<pose_slug>/<idx>.png

CLI:
  python tools/filter_pool_by_clip_margin.py \\
    --in output/pool_v1/rgba --out output/pool_v1/rgba_filtered \\
    --margin 0.10 \\
    --pairs "Soldier:civilian persons:soldier_uniform"

  --pairs is space-separated CLASS:CONFUSING_KEYWORD entries. The keyword is mapped
  to a CLIP prompt via PROMPT_MAP below (extend as needed).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROMPT_MAP = {
    "Soldier":          "a soldier in modern military combat uniform with a rifle",
    "soldier_uniform":  "a soldier in modern military combat uniform with a rifle",
    "persons":          "a civilian person in casual everyday clothes",
    "civilian":         "a civilian person in casual everyday clothes",
    "military_vehicle": "a modern military armored vehicle",
    "civilian_vehicle": "a modern civilian car",
}


def parse_pairs(spec: str) -> list[tuple[str, str]]:
    out = []
    for tok in spec.replace(",", " ").split():
        if ":" not in tok:
            continue
        cls, kw = tok.split(":", 1)
        out.append((cls.strip(), kw.strip()))
    return out


def load_clip(device: str):
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    return model, preprocess, tokenizer, torch


def encode_text(model, tokenizer, prompts: list[str], device, torch):
    with torch.no_grad():
        toks = tokenizer(prompts).to(device)
        feats = model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def encode_image(model, preprocess, img: Image.Image, device, torch):
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0).to(device)
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--pairs", required=True,
                    help='space-separated CLASS:CONFUSING entries, e.g. "Soldier:civilian persons:soldier_uniform"')
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    in_root = Path(args.in_dir); out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer, _torch = load_clip(device)

    pairs = parse_pairs(args.pairs)
    cls_to_confuse = {c: kw for c, kw in pairs}

    # Cache text features
    text_cache = {}
    for cls, kw in pairs:
        for k in (cls, kw):
            if k not in text_cache:
                prompt = PROMPT_MAP.get(k, k)
                text_cache[k] = encode_text(model, tokenizer, [prompt], device, _torch)

    n_total = 0; n_kept = 0; n_dropped = 0
    per_class_stats = {}

    for sub in sorted(in_root.iterdir()):
        if not sub.is_dir():
            continue
        # category prefix (before "__")
        if "__" in sub.name:
            cls, _ = sub.name.split("__", 1)
        else:
            cls = sub.name

        out_sub = out_root / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)

        confuse_kw = cls_to_confuse.get(cls)
        for img_path in sorted(sub.iterdir()):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            n_total += 1
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            if confuse_kw is None:
                # Class with no confusion target: pass-through copy
                shutil.copy2(img_path, out_sub / img_path.name)
                n_kept += 1
                continue

            img_feat = encode_image(model, preprocess, img, device, _torch)
            sim_cls = float((img_feat @ text_cache[cls].T).item())
            sim_conf = float((img_feat @ text_cache[confuse_kw].T).item())
            margin = sim_cls - sim_conf

            stats = per_class_stats.setdefault(cls, {"kept": 0, "dropped": 0, "margins": []})
            stats["margins"].append(margin)

            if margin >= args.margin:
                shutil.copy2(img_path, out_sub / img_path.name)
                n_kept += 1
                stats["kept"] += 1
            else:
                n_dropped += 1
                stats["dropped"] += 1

    print(f"total={n_total}  kept={n_kept}  dropped={n_dropped}")
    for cls, s in per_class_stats.items():
        m = s["margins"]
        if m:
            print(f"  {cls:18s} kept={s['kept']:4d} dropped={s['dropped']:4d}  "
                  f"margin median={float(np.median(m)):+.3f}  p10={float(np.percentile(m, 10)):+.3f}  "
                  f"p90={float(np.percentile(m, 90)):+.3f}")
    print(f"-> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
