"""
Generate pose-specific instance images with SD 1.5.

Reads:    configs/scenarios.yaml  (extracts unique (category, pose) pairs from instances spec)
Writes:   <output_dir>/<pose_slug>/{0000.png, 0001.png, ...}
          <output_dir>/poses.json  (registry of pose slug -> {category, pose_prompt, clip_scores})

The output directory structure matches what segment_methods/reseg.py expects, so the
existing segmentation + clean_pool pipeline can be run on this directory unchanged
to produce the scenario-aware instance pool.

Usage:
    python generation/gen_pose_instances.py \
        --scenarios configs/scenarios.yaml \
        --output_dir output/pose_instances \
        --samples 100 \
        --batchsize 4 \
        [--pose_prompt_template "a photo of {}, plain white background, isolated, studio lighting, full body, photorealistic"]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image

import clip


def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80]


def collect_poses(scenarios_yaml: Path) -> list[dict]:
    cfg = yaml.safe_load(scenarios_yaml.read_text())
    seen: dict[str, dict] = {}
    for s in cfg["scenarios"]:
        for inst in s.get("instances", []):
            cat = inst["category"]
            pose = inst["pose"].strip()
            slug = f"{cat}__{slugify(pose)}"
            if slug in seen:
                continue
            seen[slug] = {
                "slug": slug,
                "category": cat,
                "pose_prompt": pose,
                "scenario_ids": [s["id"]],
            }
        # also track which scenarios use each pose
    # second pass for scenario_ids
    cfg2 = yaml.safe_load(scenarios_yaml.read_text())
    for entry in seen.values():
        entry["scenario_ids"] = []
        for s in cfg2["scenarios"]:
            for inst in s.get("instances", []):
                slug = f"{inst['category']}__{slugify(inst['pose'].strip())}"
                if slug == entry["slug"] and s["id"] not in entry["scenario_ids"]:
                    entry["scenario_ids"].append(s["id"])
    return list(seen.values())


def load_sd15(device: torch.device) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.to(device)
    return pipe


def save_images(images: list[Image.Image], out_dir: Path, offset: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, img in enumerate(images):
        name = f"{offset + i:04d}.png"
        img.save(out_dir / name)
        names.append(name)
    return names


@torch.no_grad()
def clip_score_batch(clip_model, preprocess, device, images: list[Image.Image], text: str) -> list[float]:
    img_tensor = torch.stack([preprocess(im) for im in images], 0).to(device)
    text_token = clip.tokenize([text]).to(device)
    _, logits_per_text = clip_model(img_tensor, text_token)
    return logits_per_text.view(-1).cpu().tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--samples", type=int, default=100, help="images per pose")
    ap.add_argument("--batchsize", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument(
        "--pose_prompt_template", default="a photo of {}, plain white background, isolated, studio lighting, full body, photorealistic",
        help="template applied to the pose string from scenarios.yaml",
    )
    ap.add_argument("--negative_prompt", default="cartoon, anime, lowres, blurry, deformed, watermark, text, multiple")
    ap.add_argument("--resume", action="store_true", help="skip poses whose folder already has >= samples images")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poses = collect_poses(Path(args.scenarios))
    print(f"Unique poses: {len(poses)}")
    for p in poses:
        print(f"  {p['slug']:60s}  scenarios={p['scenario_ids']}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading SD 1.5 on {device}...")
    pipe = load_sd15(device)
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    registry = {}
    registry_path = out_root / "poses.json"
    results_path = out_root / "results.json"  # reseg.py compatibility
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())

    def write_results_json() -> None:
        # reseg.py expects: [{name, id, clip_scores, ...}] at <input_dir>/results.json
        items = []
        for i, p_ in enumerate(poses, start=1):
            slug_ = p_["slug"]
            entry_ = registry.get(slug_, {})
            items.append({
                "name": slug_,
                "id": i,
                "category": p_["category"],
                "pose_prompt": p_["pose_prompt"],
                "clip_scores": entry_.get("clip_scores", []),
            })
        results_path.write_text(json.dumps(items, indent=2, ensure_ascii=False))

    for p in poses:
        slug = p["slug"]
        out_dir = out_root / slug
        text = args.pose_prompt_template.format(p["pose_prompt"])
        prior = sorted(out_dir.glob("*.png")) if out_dir.exists() else []
        if args.resume and len(prior) >= args.samples:
            print(f"[skip] {slug} ({len(prior)}/{args.samples})")
            registry.setdefault(slug, {**p, "prompt": text, "clip_scores": []})
            continue

        print(f"\n[gen] {slug}\n      prompt: {text}")
        offset = len(prior)
        scores: list[float] = list(registry.get(slug, {}).get("clip_scores", []))
        while offset < args.samples:
            n = min(args.batchsize, args.samples - offset)
            out = pipe(
                prompt=[text] * n,
                negative_prompt=[args.negative_prompt] * n,
                num_inference_steps=args.steps, guidance_scale=args.guidance,
                height=args.image_size, width=args.image_size,
            )
            images = out.images
            save_images(images, out_dir, offset)
            scores.extend(clip_score_batch(clip_model, preprocess, device, images, text))
            offset += n
            print(f"  [{offset}/{args.samples}] saved")

        registry[slug] = {**p, "prompt": text, "clip_scores": scores}
        registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
        write_results_json()

    write_results_json()  # ensure final state, even if all skipped via --resume
    print(f"\nDone. Registry: {registry_path}")
    print(f"      reseg.py-compatible: {results_path}")
    print(
        "\nNext: run segmentation on the output directory:\n"
        f"  python segment_methods/reseg.py --input_dir {out_root} --output_dir <seg_out> --seg_method U2Net\n"
        "  python segment_methods/clean_pool.py --input_dir <seg_out> --image_dir "
        f"{out_root} --output_file <pool.json> --min_clip 21 --min_area 0.05 --max_area 0.95"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
