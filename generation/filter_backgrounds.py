"""
Filter SDXL backgrounds that leak unwanted objects (person, car, truck, bus, ...).

SDXL's negative prompt does not always succeed; some backgrounds end up containing
faint vehicles or pedestrians. Since these would become unlabeled ground truth
during training (false negatives), we detect and remove them before composition.

Uses HuggingFace DETR (facebook/detr-resnet-50) — pretrained on COCO. Detected
unwanted classes above a confidence + area threshold cause the background to be
moved to a "rejected" directory (kept for inspection, not deleted).

Usage:
    python generation/filter_backgrounds.py \
        --input_dir output/scenario_backgrounds \
        --rejected_dir output/scenario_backgrounds_rejected \
        --score_threshold 0.55 \
        --min_area_frac 0.005
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image


# COCO class names that count as "unwanted" in object-free military backgrounds.
UNWANTED_COCO_LABELS = {
    "person", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "boat", "airplane",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="output/scenario_backgrounds/")
    ap.add_argument("--rejected_dir", required=True, help="rejected backgrounds moved here")
    ap.add_argument("--score_threshold", type=float, default=0.55,
                    help="DETR confidence threshold to count as detection")
    ap.add_argument("--min_area_frac", type=float, default=0.005,
                    help="ignore detections smaller than this fraction of image (~0.5%)")
    ap.add_argument("--report", default="output/scenario_backgrounds_filter_report.json")
    ap.add_argument("--dry_run", action="store_true", help="report only, do not move files")
    args = ap.parse_args()

    from transformers import DetrForObjectDetection, DetrImageProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DETR on {device}...")
    proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device).eval()
    id2label = model.config.id2label

    in_root = Path(args.input_dir)
    rej_root = Path(args.rejected_dir)
    rej_root.mkdir(parents=True, exist_ok=True)

    report = []
    kept = 0
    rejected = 0
    for scenario_dir in sorted(in_root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        scen = scenario_dir.name
        for img_path in sorted(scenario_dir.glob("*.png")):
            img = Image.open(img_path).convert("RGB")
            W, H = img.size

            with torch.inference_mode():
                inputs = proc(images=img, return_tensors="pt").to(device)
                outputs = model(**inputs)
                target_sizes = torch.tensor([(H, W)]).to(device)
                results = proc.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=args.score_threshold,
                )[0]

            hits = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                cls = id2label[int(label.item())]
                if cls not in UNWANTED_COCO_LABELS:
                    continue
                x1, y1, x2, y2 = box.tolist()
                area_frac = max(0.0, (x2 - x1) * (y2 - y1)) / (W * H)
                if area_frac < args.min_area_frac:
                    continue
                hits.append({"class": cls, "score": float(score), "area_frac": area_frac})

            status = "rejected" if hits else "kept"
            entry = {"file": f"{scen}/{img_path.name}", "status": status, "hits": hits}
            report.append(entry)

            if status == "rejected":
                rejected += 1
                if not args.dry_run:
                    target = rej_root / scen / img_path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(img_path), str(target))
                hit_summary = ", ".join(f"{h['class']}({h['score']:.2f})" for h in hits)
                print(f"  [REJ] {scen}/{img_path.name}: {hit_summary}")
            else:
                kept += 1

        print(f"  -- {scen} done (running totals: kept={kept}, rejected={rejected})")

    rep_path = Path(args.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nTotal: kept={kept}  rejected={rejected}  ({rejected/max(1,kept+rejected):.1%})")
    print(f"Report: {rep_path}")
    if args.dry_run:
        print("(dry run -- no files moved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
