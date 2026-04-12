"""
방법 1: Stable Diffusion으로 넓은 장면 안에 작은 객체를 직접 생성
SD 1.5는 1920x1080을 직접 생성하기 어려우므로 1024x576(16:9)으로 생성 후 리사이즈
"""
import torch
import os
import argparse
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='LVIS_gen_FG_military_scene')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    # SD 1.5 로드
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        print("xformers not available, using default attention")
    pipe.to(device)

    # 카테고리별 프롬프트 (멀리서 본 작은 객체를 유도)
    categories = {
        "tank": [
            "a vast empty desert landscape with a single very tiny tank in the distance, aerial view, wide angle",
            "a wide open battlefield seen from far above, a single small tank barely visible, satellite view",
            "a huge empty field with one very small military tank far away, drone photography, wide shot",
            "an expansive plain with a tiny armored tank in the distance, birds eye view, miniature looking",
            "a wide aerial photograph of terrain with a single small tank, very far away, top down view",
        ],
        "soldier": [
            "a vast open field with a single tiny soldier standing far away, aerial view, wide angle shot",
            "a wide landscape seen from above with one very small soldier barely visible, drone photography",
            "an expansive terrain with a tiny military soldier in the distance, birds eye view, wide shot",
            "a huge empty ground with one very small person in military uniform far away, satellite view",
            "a wide aerial photograph of open land with a single small soldier, very distant, top down view",
        ],
        "fighter_craft": [
            "a vast blue sky with a single very tiny fighter jet in the distance, wide angle, minimal",
            "a wide open sky photograph with one very small military aircraft barely visible, far away",
            "an expansive clear sky with a tiny fighter plane in the distance, wide shot, minimalist",
            "a huge empty sky with one very small jet fighter far away, aerial photography, wide angle",
            "a wide photograph of open sky with a single small fighter aircraft, very distant, minimal",
        ],
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for cat_name, prompts in categories.items():
        cat_dir = os.path.join(args.output_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)

        for i in range(args.samples):
            prompt = prompts[i % len(prompts)]
            print(f"[{cat_name}] {i+1}/{args.samples}: {prompt[:60]}...")

            # 1024x576 (16:9)로 생성
            with torch.no_grad():
                result = pipe(
                    prompt,
                    negative_prompt="large object, close up, zoomed in, cropped, blurry",
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=576,
                    width=1024,
                ).images[0]

            # 1920x1080으로 리사이즈
            result_resized = result.resize((1920, 1080), Image.LANCZOS)

            # 원본(1024x576)과 리사이즈(1920x1080) 둘 다 저장
            result.save(os.path.join(cat_dir, f"{i:04d}_1024x576.png"))
            result_resized.save(os.path.join(cat_dir, f"{i:04d}_1920x1080.png"))
            print(f"  -> saved {cat_name}/{i:04d}")

    print(f"\nDone! Results in {args.output_dir}/")
    print("Categories:", list(categories.keys()))

if __name__ == "__main__":
    main()
