"""
Generate scenario backgrounds with SDXL using prompts from gen_scenario_prompts.py cache.

Reads:    generation/cache/scenario_prompts.json   (output of gen_scenario_prompts.py)
Reads:    configs/scenarios.yaml                   (for global image_size and scenario list)
Writes:   <output_dir>/<scenario_name>/{0000.png, 0001.png, ...}
          <output_dir>/<scenario_name>/metadata.json   (per-image: prompt index, seed, etc.)

For each scenario, we distribute `n_backgrounds` images across the cached prompts
(default 4 prompts × 4 images = 16 backgrounds). Same diffusion seed family per
scenario gives reproducibility.

Usage:
    python generation/gen_singleshot_scenes.py \
        --prompts generation/cache/scenario_prompts.json \
        --scenarios configs/scenarios.yaml \
        --output_dir output/scenario_backgrounds \
        [--model stabilityai/stable-diffusion-xl-base-1.0] \
        [--use_refiner] \
        [--steps 30] \
        [--guidance 7.0] \
        [--only 1,3,9]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from PIL import Image


def round_to_8(x: int) -> int:
    return max(8, (x // 8) * 8)


def load_pipelines(model_id: str, refiner_id: str | None, device: torch.device, dtype: torch.dtype):
    base = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)
    base.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)
    try:
        base.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    base.to(device)

    refiner = None
    if refiner_id:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_id, torch_dtype=dtype, use_safetensors=True,
            text_encoder_2=base.text_encoder_2, vae=base.vae,
        )
        try:
            refiner.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        refiner.to(device)
    return base, refiner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True, help="JSON cache from gen_scenario_prompts.py")
    ap.add_argument("--scenarios", required=True, help="configs/scenarios.yaml (for image_size)")
    ap.add_argument("--output_dir", required=True, help="root output directory")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--refiner", default="stabilityai/stable-diffusion-xl-refiner-1.0")
    ap.add_argument("--use_refiner", action="store_true", help="run SDXL refiner after base")
    ap.add_argument("--steps", type=int, default=30, help="diffusion steps for base")
    ap.add_argument("--refiner_steps", type=int, default=15)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42, help="base RNG seed; per-image seed = base_seed + index")
    ap.add_argument("--only", default="", help="comma-separated scenario ids to process")
    ap.add_argument("--n_backgrounds", type=int, default=0,
                    help="override n_backgrounds from scenarios.yaml (0 = use yaml value)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    cfg = yaml.safe_load(Path(args.scenarios).read_text())
    img_w, img_h = cfg["global"]["image_size"]
    img_w, img_h = round_to_8(img_w), round_to_8(img_h)

    prompts_cache = json.loads(Path(args.prompts).read_text())
    only_ids = {int(x) for x in args.only.split(",") if x.strip()} if args.only else set()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SDXL base ({args.model}) on {device}...")
    base, refiner = load_pipelines(
        args.model, args.refiner if args.use_refiner else None, device, dtype,
    )
    print(f"Resolution: {img_w}x{img_h} | refiner: {bool(refiner)}")

    for sid_str, entry in prompts_cache.items():
        sid = int(sid_str)
        if only_ids and sid not in only_ids:
            continue
        scenario_name = entry["name"]
        out_dir = output_dir / scenario_name
        out_dir.mkdir(parents=True, exist_ok=True)

        prompts = entry["prompts"]
        negative = entry.get("negative_prompt", "")
        n_total = args.n_backgrounds or entry.get("n_backgrounds", 16)
        per_prompt = max(1, n_total // len(prompts))
        actual_total = per_prompt * len(prompts)

        print(f"\n[{sid}] {scenario_name}: {len(prompts)} prompts x {per_prompt} = {actual_total} images")

        meta = []
        idx = 0
        for pi, prompt in enumerate(prompts):
            for k in range(per_prompt):
                seed = args.seed + sid * 1000 + pi * 100 + k
                gen = torch.Generator(device=device).manual_seed(seed)

                base_kwargs = dict(
                    prompt=prompt, negative_prompt=negative,
                    width=img_w, height=img_h,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=gen,
                )
                if refiner:
                    latent = base(**base_kwargs, output_type="latent").images
                    image = refiner(
                        prompt=prompt, negative_prompt=negative,
                        image=latent, num_inference_steps=args.refiner_steps,
                        generator=gen,
                    ).images[0]
                else:
                    image = base(**base_kwargs).images[0]

                fname = f"{idx:04d}.png"
                image.save(out_dir / fname)
                meta.append({
                    "file": fname, "prompt_index": pi, "seed": seed,
                    "prompt": prompt, "negative_prompt": negative,
                })
                idx += 1
                print(f"  [{idx}/{actual_total}] {fname}  prompt#{pi} seed={seed}")

        (out_dir / "metadata.json").write_text(
            json.dumps({"scenario_id": sid, "name": scenario_name, "images": meta}, indent=2, ensure_ascii=False)
        )

    print(f"\nDone. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
