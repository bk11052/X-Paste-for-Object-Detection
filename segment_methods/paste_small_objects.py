"""
Small object copy-paste 스크립트
전경 객체를 segmentation mask로 잘라내어 배경 이미지에 작은 크기로 paste한다.

사용법:
    python segment_methods/paste_small_objects.py \
        --fg_dir output/Military_gen_FG \
        --seg_dir output/Military_seg_FG \
        --bg_dir output/Military_BG \
        --output_dir output/Military_paste \
        --output_label output/military_paste_labels.json
"""
import os
import json
import argparse
import cv2
import numpy as np
from glob import glob
import random


# 카테고리별 크기 설정 (위성/드론 시점 기준, pixels)
# soldier < tank < fighter craft
CATEGORY_SIZE = {
    "soldier": {"min": 8, "max": 15},       # 사람은 가장 작음
    "tank": {"min": 20, "max": 35},          # 전차는 중간
    "fighter aircraft": {"min": 35, "max": 55},  # 전투기는 가장 큼
    "fighter_craft": {"min": 35, "max": 55},
    "fighter_jet": {"min": 35, "max": 55},
}

# 카테고리별 배치 패턴 (군사 현실성 반영)
CATEGORY_PLACEMENT = {
    "soldier": {
        "count": (8, 15),        # 분대~소대 규모
        "formation": "cluster",  # 밀집 대형
        "spacing": (5, 15),      # 병사 간 간격 (px)
    },
    "tank": {
        "count": (3, 6),         # 소대 규모
        "formation": "line",     # 종대/횡대
        "spacing": (30, 60),     # 전차 간 간격
    },
    "fighter aircraft": {
        "count": (2, 4),         # 편대 규모
        "formation": "line",     # 활주로 정렬
        "spacing": (50, 80),
    },
    "fighter_craft": {
        "count": (2, 4),
        "formation": "line",
        "spacing": (50, 80),
    },
    "fighter_jet": {
        "count": (2, 4),
        "formation": "line",
        "spacing": (50, 80),
    },
}

DEFAULT_SIZE = {"min": 15, "max": 30}
DEFAULT_PLACEMENT = {"count": (3, 8), "formation": "random", "spacing": (20, 40)}


def load_foreground(img_path, mask_path, target_size):
    """전경 이미지를 mask로 잘라서 target_size로 리사이즈"""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None, None

    # mask로 전경 영역 crop
    mask_bin = (mask > 128).astype(np.uint8)
    coords = np.where(mask_bin > 0)
    if len(coords[0]) == 0:
        return None, None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    fg_crop = img[y_min:y_max+1, x_min:x_max+1]
    mask_crop = mask_bin[y_min:y_max+1, x_min:x_max+1]

    # aspect ratio 유지하면서 리사이즈
    h, w = fg_crop.shape[:2]
    aspect = w / max(h, 1)
    if aspect >= 1:
        new_w = target_size
        new_h = max(int(target_size / aspect), 1)
    else:
        new_h = target_size
        new_w = max(int(target_size * aspect), 1)

    fg_resized = cv2.resize(fg_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    return fg_resized, mask_resized


def generate_positions(bg_h, bg_w, count, formation, spacing_range, obj_size, margin=50):
    """대형에 따른 paste 위치 생성"""
    positions = []
    min_sp, max_sp = spacing_range

    if formation == "cluster":
        # 밀집 대형: 중심점 잡고 주변에 랜덤 배치
        cx = random.randint(margin, bg_w - margin)
        cy = random.randint(margin, bg_h - margin)
        for _ in range(count):
            dx = random.randint(-max_sp * 2, max_sp * 2)
            dy = random.randint(-max_sp * 2, max_sp * 2)
            x = max(margin, min(bg_w - margin - obj_size, cx + dx))
            y = max(margin, min(bg_h - margin - obj_size, cy + dy))
            positions.append((x, y))

    elif formation == "line":
        # 종대/횡대: 가로 또는 세로로 일렬 배치
        horizontal = random.random() > 0.5
        start_x = random.randint(margin, bg_w // 3)
        start_y = random.randint(margin, bg_h // 3)
        angle = random.uniform(-0.15, 0.15)  # 약간의 각도 변화

        for i in range(count):
            spacing = random.randint(min_sp, max_sp)
            if horizontal:
                x = start_x + i * spacing
                y = start_y + int(i * spacing * angle) + random.randint(-5, 5)
            else:
                x = start_x + int(i * spacing * angle) + random.randint(-5, 5)
                y = start_y + i * spacing
            x = max(margin, min(bg_w - margin - obj_size, x))
            y = max(margin, min(bg_h - margin - obj_size, y))
            positions.append((x, y))

    else:
        # 랜덤 배치
        for _ in range(count):
            x = random.randint(margin, bg_w - margin - obj_size)
            y = random.randint(margin, bg_h - margin - obj_size)
            positions.append((x, y))

    return positions


def paste_object(bg, fg, mask, x, y):
    """배경에 전경을 alpha blending으로 paste"""
    fh, fw = fg.shape[:2]
    bh, bw = bg.shape[:2]

    # 경계 체크
    if x + fw > bw or y + fh > bh or x < 0 or y < 0:
        return False, None

    mask_3ch = np.stack([mask] * 3, axis=-1).astype(np.float32)
    # 가장자리 살짝 블러 (자연스러운 합성)
    mask_3ch = cv2.GaussianBlur(mask_3ch, (3, 3), 0.5)

    roi = bg[y:y+fh, x:x+fw].astype(np.float32)
    fg_f = fg[:, :, :3].astype(np.float32) if fg.shape[2] >= 3 else fg.astype(np.float32)

    blended = roi * (1 - mask_3ch) + fg_f * mask_3ch
    bg[y:y+fh, x:x+fw] = blended.astype(np.uint8)

    bbox = [int(x), int(y), int(fw), int(fh)]
    return True, bbox


def main():
    parser = argparse.ArgumentParser(description='Small object copy-paste')
    parser.add_argument('--fg_dir', type=str, required=True, help='전경 이미지 디렉토리')
    parser.add_argument('--seg_dir', type=str, required=True, help='segmentation mask 디렉토리')
    parser.add_argument('--bg_dir', type=str, required=True, help='배경 이미지 디렉토리')
    parser.add_argument('--output_dir', type=str, required=True, help='합성 이미지 출력')
    parser.add_argument('--output_label', type=str, default=None, help='COCO JSON label 출력')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 배경 이미지 로드 (1920x1080만)
    bg_files = sorted(glob(os.path.join(args.bg_dir, '*1920x1080*')))
    if not bg_files:
        bg_files = sorted(glob(os.path.join(args.bg_dir, '*.png')) + glob(os.path.join(args.bg_dir, '*.jpg')))
    print(f"Backgrounds: {len(bg_files)}")

    # 카테고리별 전경 + mask 수집
    categories = {}
    cat_dirs = sorted([d for d in os.listdir(args.fg_dir) if os.path.isdir(os.path.join(args.fg_dir, d))])

    for cat_name in cat_dirs:
        fg_files = sorted(glob(os.path.join(args.fg_dir, cat_name, '*.png')))
        pairs = []
        for fg_path in fg_files:
            fname = os.path.basename(fg_path)
            mask_path = os.path.join(args.seg_dir, cat_name, fname)
            if os.path.exists(mask_path):
                pairs.append((fg_path, mask_path))
        if pairs:
            categories[cat_name] = pairs
            print(f"  [{cat_name}] {len(pairs)} fg+mask pairs")

    # COCO JSON 준비
    coco = {"images": [], "annotations": [], "categories": []}
    for cat_id, cat_name in enumerate(sorted(categories.keys()), start=1):
        coco["categories"].append({"id": cat_id, "name": cat_name, "supercategory": "military"})
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}

    image_id = 0
    ann_id = 0

    # 각 배경에 각 카테고리를 paste
    for bg_path in bg_files:
        bg_name = os.path.splitext(os.path.basename(bg_path))[0]
        bg_orig = cv2.imread(bg_path)
        if bg_orig is None:
            continue
        bg_h, bg_w = bg_orig.shape[:2]
        print(f"\nBackground: {bg_name} ({bg_w}x{bg_h})")

        for cat_name, pairs in categories.items():
            bg = bg_orig.copy()
            size_cfg = CATEGORY_SIZE.get(cat_name, DEFAULT_SIZE)
            place_cfg = CATEGORY_PLACEMENT.get(cat_name, DEFAULT_PLACEMENT)

            count = random.randint(*place_cfg["count"])
            avg_size = random.randint(size_cfg["min"], size_cfg["max"])

            positions = generate_positions(
                bg_h, bg_w, count,
                place_cfg["formation"],
                place_cfg["spacing"],
                avg_size
            )

            out_name = f"{bg_name}_{cat_name}.png"
            print(f"  [{cat_name}] pasting {count} objects (size ~{avg_size}px, formation: {place_cfg['formation']})")

            pasted = 0
            for pos_x, pos_y in positions:
                fg_path, mask_path = random.choice(pairs)
                # 개체마다 약간의 크기 변동
                obj_size = avg_size + random.randint(-3, 3)
                obj_size = max(size_cfg["min"], min(size_cfg["max"], obj_size))

                fg, mask = load_foreground(fg_path, mask_path, obj_size)
                if fg is None:
                    continue

                # 랜덤 회전 (위성 시점이므로 360도 회전 자연스러움)
                angle = random.randint(0, 359)
                M = cv2.getRotationMatrix2D((fg.shape[1]//2, fg.shape[0]//2), angle, 1.0)
                fg = cv2.warpAffine(fg, M, (fg.shape[1], fg.shape[0]))
                mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

                success, bbox = paste_object(bg, fg, mask, pos_x, pos_y)
                if success:
                    coco["annotations"].append({
                        "id": ann_id, "image_id": image_id,
                        "category_id": cat_name_to_id[cat_name],
                        "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0
                    })
                    ann_id += 1
                    pasted += 1

            # 저장
            out_path = os.path.join(args.output_dir, out_name)
            cv2.imwrite(out_path, bg)

            coco["images"].append({
                "id": image_id,
                "file_name": out_name,
                "width": bg_w, "height": bg_h
            })
            image_id += 1
            print(f"    -> {pasted} objects pasted, saved {out_name}")

    # COCO JSON 저장
    if args.output_label:
        os.makedirs(os.path.dirname(args.output_label) or '.', exist_ok=True)
        with open(args.output_label, 'w') as f:
            json.dump(coco, f, indent=2)
        print(f"\nLabels saved: {args.output_label}")

    print(f"\nDone!")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")


if __name__ == '__main__':
    main()
