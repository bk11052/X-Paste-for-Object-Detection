"""
군사/커스텀 카테고리 이미지 segmentation 스크립트
rembg (U2Net 기반)를 사용하여 전경 mask를 추출한다.

사용법:
    pip install rembg onnxruntime
    python segment_methods/seg_military.py \
        --input_dir output/Military_gen_FG \
        --output_dir output/Military_seg_FG

rembg가 없으면 GrabCut fallback 사용.
"""
import os
import argparse
import cv2
import numpy as np
from glob import glob

try:
    from rembg import remove as rembg_remove
    from PIL import Image
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False


def segment_rembg(img_bgr):
    """rembg (U2Net) 기반 segmentation"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    # rembg는 RGBA 반환, alpha 채널이 mask
    result = rembg_remove(pil_img)
    alpha = np.array(result)[:, :, 3]
    return (alpha > 128).astype(np.uint8) * 255


def segment_grabcut(img_bgr):
    """GrabCut fallback"""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    margin = max(int(min(h, w) * 0.05), 5)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return fg_mask


def main():
    parser = argparse.ArgumentParser(description='커스텀 카테고리 이미지 segmentation')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='생성 이미지 디렉토리 (카테고리별 서브폴더)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='mask 출력 디렉토리')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'rembg', 'grabcut'],
                        help='segmentation 방법 (auto: rembg 우선, 없으면 grabcut)')
    args = parser.parse_args()

    # method 결정
    if args.method == 'auto':
        use_rembg = HAS_REMBG
    elif args.method == 'rembg':
        if not HAS_REMBG:
            print("rembg not installed. Run: pip install rembg onnxruntime")
            return
        use_rembg = True
    else:
        use_rembg = False

    print(f"Segmentation method: {'rembg (U2Net)' if use_rembg else 'GrabCut'}")
    segment_fn = segment_rembg if use_rembg else segment_grabcut

    # 카테고리 폴더 순회
    cat_dirs = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])

    print(f"Categories: {cat_dirs}")

    for cat_name in cat_dirs:
        cat_input = os.path.join(args.input_dir, cat_name)
        cat_output = os.path.join(args.output_dir, cat_name)
        os.makedirs(cat_output, exist_ok=True)

        img_files = sorted(glob(os.path.join(cat_input, '*.png')) + glob(os.path.join(cat_input, '*.jpg')))
        print(f"  [{cat_name}] {len(img_files)} images")

        skipped = 0
        processed = 0
        for img_path in img_files:
            filename = os.path.basename(img_path)
            out_path = os.path.join(cat_output, filename)

            # 이미 처리된 파일은 건너뛰기
            if os.path.exists(out_path):
                skipped += 1
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            mask = segment_fn(img_bgr)
            cv2.imwrite(out_path, mask)
            processed += 1

        print(f"    -> {processed} processed, {skipped} skipped (already exists)")

    print(f"\nDone! Masks saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
