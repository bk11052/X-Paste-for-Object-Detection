"""
Bbox 시각화 스크립트
COCO JSON의 bbox를 이미지 위에 그려서 저장한다.

사용법:
    python segment_methods/visualize_bbox.py \
        --input_dir output/LVIS_gen_FG \
        --label_file output/bbox_labels.json \
        --output_dir output/bbox_vis \
        --max_images 5
"""
import os
import json
import argparse
import cv2
import numpy as np


# 카테고리별 색상 (BGR)
COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (128, 0, 255),
]


def main():
    parser = argparse.ArgumentParser(description='Bbox 시각화')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='이미지 루트 디렉토리')
    parser.add_argument('--label_file', type=str, required=True,
                        help='COCO JSON 파일 경로')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='시각화 결과 저장 디렉토리')
    parser.add_argument('--max_images', type=int, default=0,
                        help='카테고리별 최대 시각화 수 (0=전체)')
    args = parser.parse_args()

    with open(args.label_file) as f:
        coco = json.load(f)

    # 인덱스 구축
    cat_map = {c['id']: c['name'] for c in coco['categories']}
    img_map = {img['id']: img for img in coco['images']}

    # image_id별 annotation 그룹
    ann_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)

    os.makedirs(args.output_dir, exist_ok=True)

    # 카테고리별 카운터
    cat_count = {}
    drawn = 0

    for img_id, anns in ann_by_image.items():
        img_info = img_map[img_id]
        cat_id = anns[0]['category_id']
        cat_name = cat_map[cat_id]

        # max_images 제한
        if args.max_images > 0:
            cat_count[cat_name] = cat_count.get(cat_name, 0) + 1
            if cat_count[cat_name] > args.max_images:
                continue

        img_path = os.path.join(args.input_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue

        for ann in anns:
            x, y, w, h = ann['bbox']
            color = COLORS[(ann['category_id'] - 1) % len(COLORS)]
            label = cat_map[ann['category_id']]

            # bbox 그리기
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

            # label 텍스트
            label_text = f"{label} ({w}x{h})"
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img, (int(x), int(y) - th - 6), (int(x) + tw + 4, int(y)), color, -1)
            cv2.putText(img, label_text, (int(x) + 2, int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # 저장
        out_name = img_info['file_name'].replace('/', '_')
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, img)
        drawn += 1

    print(f"Done! {drawn} images saved to {args.output_dir}/")
    print(f"Categories: {list(cat_count.keys()) if cat_count else list(cat_map.values())}")


if __name__ == '__main__':
    main()
