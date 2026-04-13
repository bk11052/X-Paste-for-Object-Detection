"""
군사 도메인 배경 이미지 생성
위성/드론 시점의 배경을 SD 1.5로 생성한다.
객체가 없는 순수 배경만 생성하여 copy-paste 용도로 사용.

사용법:
    python generation/gen_military_bg.py \
        --output_dir output/Military_BG \
        --samples 3 --gpu 0
"""
import torch
import os
import argparse
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output/Military_BG')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        print("xformers not available, using default attention")
    pipe.to(device)

    # 전략: 3가지 다른 군사 환경 배경
    # - 객체를 명시적으로 배제하는 negative prompt 사용
    # - 위성/드론 시점 (top-down)으로 통일 → small object paste와 시점 일관성
    # - 1024x576 (16:9) 생성 후 1920x1080 리사이즈
    backgrounds = [
        {
            "name": "desert_base",
            "prompt": "satellite view of empty desert terrain with dirt roads, "
                      "military base area, flat sandy ground, no vehicles, "
                      "no people, aerial photography, top down view, high resolution",
            "negative": "vehicles, tanks, people, soldiers, aircraft, buildings, text, watermark",
        },
        {
            "name": "airfield",
            "prompt": "drone photograph of empty military airfield runway, "
                      "concrete tarmac, grass field beside runway, "
                      "no aircraft, no vehicles, no people, "
                      "aerial view, top down, clear weather",
            "negative": "planes, jets, vehicles, people, text, watermark, blurry",
        },
        {
            "name": "open_field",
            "prompt": "satellite image of open green field with dirt paths, "
                      "rural terrain, farmland, no buildings, no vehicles, "
                      "no people, birds eye view, top down photograph",
            "negative": "buildings, vehicles, people, urban, city, text, watermark",
        },
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    for i, bg in enumerate(backgrounds):
        if i >= args.samples:
            break

        print(f"[{i+1}/{args.samples}] Generating: {bg['name']}")
        print(f"  Prompt: {bg['prompt'][:80]}...")

        with torch.no_grad():
            result = pipe(
                bg["prompt"],
                negative_prompt=bg["negative"],
                num_inference_steps=50,
                guidance_scale=7.5,
                height=576,
                width=1024,
            ).images[0]

        # 1920x1080으로 리사이즈
        result_hd = result.resize((1920, 1080), Image.LANCZOS)

        result.save(os.path.join(args.output_dir, f"{bg['name']}_1024x576.png"))
        result_hd.save(os.path.join(args.output_dir, f"{bg['name']}_1920x1080.png"))
        print(f"  -> saved {bg['name']}")

    print(f"\nDone! {args.samples} backgrounds in {args.output_dir}/")


if __name__ == "__main__":
    main()
