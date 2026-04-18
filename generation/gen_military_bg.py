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

    NEGATIVE = "vehicles, tanks, people, soldiers, aircraft, text, watermark, blurry"

    backgrounds = [
        {"name": "desert", "prompt": "satellite view of empty desert terrain with dirt roads, flat sandy ground, aerial photography, top down view, high resolution"},
        {"name": "airfield", "prompt": "drone photograph of empty military airfield runway, concrete tarmac, grass field beside runway, aerial view, top down, clear weather"},
        {"name": "open_field", "prompt": "satellite image of open green field with dirt paths, rural terrain, farmland, birds eye view, top down photograph"},
        {"name": "forest", "prompt": "aerial view of forest clearing with trees around edges, dirt ground, top down satellite photograph, high resolution"},
        {"name": "urban", "prompt": "satellite view of empty urban road intersection, asphalt streets, sidewalks, top down aerial photograph, no cars, no people"},
        {"name": "mountain", "prompt": "drone view of rocky mountain terrain, gravel paths, sparse vegetation, aerial top down photograph, clear day"},
        {"name": "snow", "prompt": "satellite view of snowy flat terrain, winter landscape, white ground, dirt road visible, top down aerial photograph"},
        {"name": "coastal", "prompt": "aerial view of sandy coastal area, beach terrain, flat ground near shoreline, top down satellite photograph"},
        {"name": "grassland", "prompt": "satellite view of wide grassland plain, green meadow, scattered dirt trails, aerial top down photograph"},
        {"name": "muddy_road", "prompt": "drone view of muddy unpaved road through countryside, wet terrain, puddles, aerial top down photograph"},
        {"name": "highway", "prompt": "satellite view of empty highway road, asphalt surface, lane markings, roadside, top down aerial photograph, no cars"},
        {"name": "industrial", "prompt": "aerial view of empty industrial yard, concrete ground, warehouse area, top down satellite photograph, no vehicles"},
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    for bg in backgrounds:
        for s in range(args.samples):
            print(f"[{total+1}] Generating: {bg['name']}_{s}")

            with torch.no_grad():
                result = pipe(
                    bg["prompt"],
                    negative_prompt=NEGATIVE,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=640,
                    width=640,
                ).images[0]

            result.save(os.path.join(args.output_dir, f"{bg['name']}_{s:03d}.png"))
            total += 1

    print(f"\nDone! {total} backgrounds in {args.output_dir}/")


if __name__ == "__main__":
    main()
