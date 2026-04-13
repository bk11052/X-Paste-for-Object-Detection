"""
Bbox 단위 label 생성 스크립트
생성된 전경 이미지에서 배경을 제거하고, bbox를 추출하여 COCO JSON 형식으로 출력한다.

사용법:
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
from PIL import Image
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
    """단순 threshold 기반 배경 제거 (SD 생성 이미지는 대부분 밝은/단색 배경)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu threshold로 전경/배경 분리
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # morphological 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask > 0).astype(np.uint8)


def extract_mask_grabcut(img_bgr):
    """GrabCut 기반 배경 제거 (더 정확하지만 느림)"""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    # 이미지 가장자리를 배경으로 가정, 중앙을 전경으로
    margin = max(int(min(h, w) * 0.05), 5)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    # 전경 + 가능한 전경을 mask로
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return fg_mask


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
    parser.add_argument('--output_file', type=str, required=True,
                        help='출력 COCO JSON 파일 경로')
    parser.add_argument('--method', type=str, default='threshold',
                        choices=['threshold', 'grabcut'],
                        help='배경 제거 방법: threshold (빠름) 또는 grabcut (정확)')
    args = parser.parse_args()

    # 카테고리 수집 (results.json이 있으면 사용, 없으면 폴더명에서)
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

    print(f"Categories: {cat_names}")
    print(f"Method: {args.method}")

    extract_mask = extract_mask_threshold if args.method == 'threshold' else extract_mask_grabcut

    # COCO JSON 구조
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 카테고리 등록 (1-indexed)
    for cat_id, cat_name in enumerate(cat_names, start=1):
        coco["categories"].append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "object"
        })

    image_id = 0
    ann_id = 0

    for cat_id, cat_name in enumerate(cat_names, start=1):
        cat_dir = os.path.join(args.input_dir, cat_name)
        if not os.path.isdir(cat_dir):
            print(f"  [SKIP] {cat_name}: directory not found")
            continue

        img_files = sorted(glob(os.path.join(cat_dir, '*.png')) + glob(os.path.join(cat_dir, '*.jpg')))
        print(f"  [{cat_name}] {len(img_files)} images")

        for img_path in img_files:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            rel_path = os.path.relpath(img_path, args.input_dir)

            # mask 추출
            mask = extract_mask(img_bgr)
            bbox, area = mask_to_bbox_xywh(mask)

            if bbox is None:
                continue

            # image 등록
            coco["images"].append({
                "id": image_id,
                "file_name": rel_path,
                "width": w,
                "height": h
            })

            # annotation 등록
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })

            image_id += 1
            ann_id += 1

    # 저장
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"\nDone!")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Categories: {len(coco['categories'])}")
    print(f"  Output: {args.output_file}")


if __name__ == '__main__':
    main()
