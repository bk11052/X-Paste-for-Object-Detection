"""
Bbox 단위 label 생성 스크립트
생성된 전경 이미지와 segmentation mask에서 bbox를 추출하여 COCO JSON 형식으로 출력한다.

사용법:
    # 기존 segmentation mask 사용 (권장)
    python segment_methods/gen_bbox_labels.py \
        --input_dir output/LVIS_gen_FG \
        --seg_dir output/LVIS_seg_FG \
        --output_file output/bbox_labels.json \
        --method segmask

    # threshold 기반 (seg mask 없을 때)
    python segment_methods/gen_bbox_labels.py \
        --input_dir output/LVIS_gen_FG \
        --output_file output/bbox_labels.json \
        --method threshold
"""
import os
import json
import argparse
import cv2
import numpy as np
from glob import glob


def get_largest_connect_component(img):
    """clean_pool.py에서 가져온 함수: 가장 큰 연결 컴포넌트만 남김"""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    if len(area) >= 1:
        max_idx = np.argmax(area)
        img2 = np.zeros_like(img)
        cv2.fillPoly(img2, [contours[max_idx]], 1)
        return img2
    else:
        return img


def extract_mask_threshold(img_bgr):
    """단순 threshold 기반 배경 제거"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask > 0).astype(np.uint8)


def extract_mask_grabcut(img_bgr):
    """GrabCut 기반 배경 제거"""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    margin = max(int(min(h, w) * 0.05), 5)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return fg_mask


def load_seg_mask(seg_dir, cat_name, filename, seg_methods=None):
    """기존 segmentation mask 로드. 여러 method가 있으면 가장 먼저 찾는 것을 사용."""
    if seg_methods is None:
        # seg_dir 하위 폴더를 seg method로 간주
        try:
            seg_methods = sorted([
                d for d in os.listdir(seg_dir)
                if os.path.isdir(os.path.join(seg_dir, d))
            ])
        except FileNotFoundError:
            return None

    for method in seg_methods:
        mask_path = os.path.join(seg_dir, method, cat_name, filename)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return (mask > 128).astype(np.uint8)

    # method 서브폴더 없이 직접 카테고리 폴더인 경우
    mask_path = os.path.join(seg_dir, cat_name, filename)
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return (mask > 128).astype(np.uint8)

    return None


def mask_to_bbox_xywh(mask):
    """binary mask에서 bbox [x, y, w, h] 추출 (COCO 형식)"""
    mask = get_largest_connect_component(mask)
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None, 0
    y_min, y_max = int(coords[0].min()), int(coords[0].max())
    x_min, x_max = int(coords[1].min()), int(coords[1].max())
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    if w <= 0 or h <= 0:
        return None, 0
    return [x_min, y_min, w, h], w * h


def main():
    parser = argparse.ArgumentParser(description='생성 이미지에서 bbox label을 COCO JSON으로 추출')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='생성 이미지 디렉토리 (카테고리별 서브폴더)')
    parser.add_argument('--seg_dir', type=str, default=None,
                        help='segmentation mask 디렉토리 (method=segmask일 때 필수). '
                             '구조: seg_dir/{method}/{category}/{file}.png 또는 seg_dir/{category}/{file}.png')
    parser.add_argument('--output_file', type=str, required=True,
                        help='출력 COCO JSON 파일 경로')
    parser.add_argument('--method', type=str, default='segmask',
                        choices=['segmask', 'threshold', 'grabcut'],
                        help='mask 소스: segmask (기존 seg mask 사용), threshold, grabcut')
    args = parser.parse_args()

    if args.method == 'segmask' and args.seg_dir is None:
        parser.error("--seg_dir is required when method=segmask")

    # 카테고리 수집
    results_path = os.path.join(args.input_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            cat_data = json.load(f)
        if isinstance(cat_data, list):
            cat_names = [c['name'] for c in cat_data]
        else:
            cat_names = [c['name'] for c in cat_data['categories']]
    else:
        cat_names = sorted([
            d for d in os.listdir(args.input_dir)
            if os.path.isdir(os.path.join(args.input_dir, d))
        ])

    # seg method 목록 (segmask 모드)
    seg_methods = None
    if args.method == 'segmask' and args.seg_dir:
        seg_methods = sorted([
            d for d in os.listdir(args.seg_dir)
            if os.path.isdir(os.path.join(args.seg_dir, d))
        ])
        # method 서브폴더가 카테고리 이름과 겹치면 method 없이 직접 구조
        if set(seg_methods) & set(cat_names):
            seg_methods = None
        print(f"Seg methods: {seg_methods if seg_methods else '(direct category folders)'}")

    print(f"Categories: {cat_names}")
    print(f"Method: {args.method}")

    # COCO JSON 구조
    coco = {"images": [], "annotations": [], "categories": []}

    for cat_id, cat_name in enumerate(cat_names, start=1):
        coco["categories"].append({
            "id": cat_id, "name": cat_name, "supercategory": "object"
        })

    image_id = 0
    ann_id = 0
    skipped = 0

    for cat_id, cat_name in enumerate(cat_names, start=1):
        cat_dir = os.path.join(args.input_dir, cat_name)
        if not os.path.isdir(cat_dir):
            print(f"  [SKIP] {cat_name}: directory not found")
            continue

        img_files = sorted(glob(os.path.join(cat_dir, '*.png')) + glob(os.path.join(cat_dir, '*.jpg')))
        found = 0

        for img_path in img_files:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            rel_path = os.path.relpath(img_path, args.input_dir)
            filename = os.path.basename(img_path)

            # mask 추출
            if args.method == 'segmask':
                mask = load_seg_mask(args.seg_dir, cat_name, filename, seg_methods)
                if mask is None:
                    skipped += 1
                    continue
            elif args.method == 'threshold':
                mask = extract_mask_threshold(img_bgr)
            else:
                mask = extract_mask_grabcut(img_bgr)

            bbox, area = mask_to_bbox_xywh(mask)
            if bbox is None:
                skipped += 1
                continue

            coco["images"].append({
                "id": image_id, "file_name": rel_path, "width": w, "height": h
            })
            coco["annotations"].append({
                "id": ann_id, "image_id": image_id, "category_id": cat_id,
                "bbox": bbox, "area": area, "iscrowd": 0
            })

            image_id += 1
            ann_id += 1
            found += 1

        print(f"  [{cat_name}] {found}/{len(img_files)} images (skipped {len(img_files) - found})")

    # 저장
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"\nDone!")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Categories: {len(coco['categories'])}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {args.output_file}")


if __name__ == '__main__':
    main()
