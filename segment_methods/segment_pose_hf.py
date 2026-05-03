"""
Segment pose instances with HuggingFace CLIPSeg (no third_party setup needed).

This is a simpler alternative to the multi-method pipeline:
   reseg.py + clean_pool.py + pose_pool_reorganize.py
which requires downloading clipseg/, U-2-Net/, etc. The single-method HF CLIPSeg
output is good enough for paper figures and a first round of training.

Inputs:
  --input_dir   output of generation/gen_pose_instances.py
                structure: <input_dir>/<pose_slug>/{0000.png, ...} + results.json
  --output_dir  target; will produce <output_dir>/<pose_slug>/{0000.png, ...} (RGBA)

Outputs:
  <output_dir>/<pose_slug>/<idx>.png   RGBA, cropped to mask bbox
  <output_dir>/segmentation_meta.json  per-image: clip score, mask area, status

Usage:
  python segment_methods/segment_pose_hf.py \
      --input_dir output/pose_instances \
      --output_dir output/pose_pool_by_slug \
      --min_area 0.02 --min_clip 18 --max_area 0.95
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def load_clipseg(device):
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()
    return proc, model


def load_clip_scorer(device):
    import clip as openai_clip
    model, preprocess = openai_clip.load("ViT-L/14", device=device)
    return model, preprocess, openai_clip


@torch.inference_mode()
def predict_mask(proc, model, device, img: Image.Image, text: str) -> np.ndarray:
    inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = out.logits  # (1, h, w) -- low-res
    logits = torch.nn.functional.interpolate(
        logits.unsqueeze(1), size=(img.size[1], img.size[0]), mode="bilinear", align_corners=False,
    ).squeeze().sigmoid().cpu().numpy()
    return logits  # float in [0, 1]


def largest_cc(mask_bin: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_bin
    biggest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask_bin)
    cv2.fillPoly(out, [biggest], 1)
    return out


@torch.inference_mode()
def clip_score(clip_model, clip_pre, openai_clip, device, rgba: np.ndarray, text: str) -> float:
    # composite onto white for CLIP scoring (matches X-Paste convention)
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    a = (rgba[..., 3:4].astype(np.float32) / 255.0)
    wb = rgb * a + (1.0 - a)
    pil = Image.fromarray((wb * 255).astype(np.uint8))
    img_t = clip_pre(pil).unsqueeze(0).to(device)
    text_t = openai_clip.tokenize([text]).to(device)
    _, logits_per_text = clip_model(img_t, text_t)
    return float(logits_per_text.view(-1).item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="output of gen_pose_instances.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.40, help="sigmoid threshold for binary mask")
    ap.add_argument("--min_area", type=float, default=0.02, help="discard if mask covers < this fraction of image")
    ap.add_argument("--max_area", type=float, default=0.95, help="discard if mask covers > this fraction (likely full-image)")
    ap.add_argument("--min_clip", type=float, default=18.0, help="discard if CLIP score below this (long prompts have lower scores)")
    ap.add_argument("--prompt_for_seg", default="", help="override: text prompt for CLIPSeg (default: use pose_prompt from results.json)")
    ap.add_argument("--use_category_prompt", action="store_true",
                    help="use just the category name (e.g. 'soldier', 'tank') as seg prompt instead of full pose. "
                         "More reliable across pose variations.")
    args = ap.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results_path = in_root / "results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run gen_pose_instances.py first (it writes results.json).")
        return 2
    items = json.loads(results_path.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIPSeg on {device}...")
    proc, seg_model = load_clipseg(device)
    print("Loading CLIP for scoring...")
    clip_model, clip_pre, openai_clip = load_clip_scorer(device)

    meta = {}
    kept_total = 0
    dropped_total = 0
    for it in items:
        slug = it["name"]
        if args.prompt_for_seg:
            seg_text = args.prompt_for_seg
        elif args.use_category_prompt:
            seg_text = it.get("category", slug.split("__")[0])
        else:
            seg_text = it.get("pose_prompt", slug.replace("_", " "))
        in_dir = in_root / slug
        if not in_dir.exists():
            print(f"[skip] {slug}: input dir missing")
            continue
        out_dir = out_root / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() == ".png")

        slug_meta = []
        kept = 0
        for k, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            arr = np.array(img)
            H, W = arr.shape[:2]

            prob = predict_mask(proc, seg_model, device, img, seg_text)
            mask_bin = (prob > args.threshold).astype(np.uint8)
            mask_bin = largest_cc(mask_bin)
            area_frac = float(mask_bin.sum()) / max(1, H * W)

            status = "kept"
            if area_frac < args.min_area or area_frac > args.max_area:
                status = f"area_out_of_range({area_frac:.3f})"

            ys, xs = np.where(mask_bin > 0)
            if len(xs) == 0:
                status = "empty_mask"
            if status != "kept":
                slug_meta.append({"src": f.name, "status": status, "area": area_frac, "clip": None})
                dropped_total += 1
                continue

            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            rgba = np.dstack([arr, mask_bin * 255]).astype(np.uint8)
            rgba_crop = rgba[y1:y2, x1:x2]

            score = clip_score(clip_model, clip_pre, openai_clip, device, rgba_crop, seg_text)
            if score < args.min_clip:
                slug_meta.append({"src": f.name, "status": f"low_clip({score:.2f})", "area": area_frac, "clip": score})
                dropped_total += 1
                continue

            out_path = out_dir / f"{kept:04d}.png"
            Image.fromarray(rgba_crop, mode="RGBA").save(out_path)
            slug_meta.append({"src": f.name, "out": out_path.name, "status": "kept", "area": area_frac, "clip": score})
            kept += 1

        kept_total += kept
        print(f"  [{slug}] {kept}/{len(files)} kept (text='{seg_text[:60]}...')")
        meta[slug] = slug_meta

    (out_root / "segmentation_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nDone. kept={kept_total} dropped={dropped_total}")
    print(f"  -> {out_root}")
    print(
        f"\nNext: python generation/compose_scene.py "
        f"--pose_pool_dir {out_root} "
        f"--backgrounds_dir output/scenario_backgrounds "
        f"--scenarios configs/scenarios.yaml --output_dir output/composed_train --save_viz"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
