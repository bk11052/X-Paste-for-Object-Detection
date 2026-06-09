"""
Render the single-example pipeline-tour video from captured stages.

Input  : --cap  (dir written by capture_stages.py)
Output : --out  mp4 (1920x1080, 30fps, ~23s)

Example:
  python tools/viz/render_example_tour.py --cap runs/viz/demo_stages \
      --out runs/viz/pipeline_tour.mp4
Requires: Pillow, ffmpeg. Korean font (e.g. `sudo apt install fonts-nanum`) or --font.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
BG = (12, 14, 18)
CAP_H = 168
BLUE, ORANGE, RED, PURPLE, GREEN = (66, 133, 244), (244, 160, 0), (219, 68, 95), (149, 70, 200), (15, 170, 90)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]


def load_font(sz, extra=None):
    cands = ([extra] if extra else []) + FONT_CANDIDATES
    for p in cands:
        if p and Path(p).exists():
            try:
                return ImageFont.truetype(p, sz, index=0)
            except Exception:
                pass
    print("WARNING: no Korean font found; install fonts-nanum or pass --font")
    return ImageFont.load_default()


def ease(t):
    return t * t * (3 - 2 * t)


def clamp01(x):
    return max(0.0, min(1.0, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default="demo_stages")
    ap.add_argument("--out", default="pipeline_tour.mp4")
    ap.add_argument("--font", default=None)
    args = ap.parse_args()

    CAP = Path(args.cap)
    F_TITLE = load_font(76, args.font); F_SUB = load_font(38, args.font)
    F_BANNER = load_font(38, args.font); F_CAP = load_font(46, args.font); F_TAG = load_font(32, args.font)

    meta = json.loads((CAP / "meta.json").read_text())
    HW, HH = meta["W"], meta["H"]; bbox = meta["bbox"]; tgt = meta["target"]
    host = Image.open(CAP / "host.png").convert("RGB")
    corrected = Image.open(CAP / "corrected.png").convert("RGB")
    mask = np.array(Image.open(CAP / "mask_paste.png").convert("L"))
    inst_src = Image.open(CAP / "instance_src.png").convert("RGBA")
    inst_placed = Image.open(CAP / "instance_placed.png").convert("RGBA")
    hist_inset = Image.open(CAP / "hist_inset.png").convert("RGB")
    hist_target = Image.open(CAP / "hist_target.png").convert("RGB")

    composed_before = host.copy()
    composed_before.paste(inst_placed, (bbox[0], bbox[1]), inst_placed)

    STAGE_TOP, STAGE_BOT = 26, H - CAP_H - 14
    avail_h = STAGE_BOT - STAGE_TOP
    scale = min((W - 360) / HW, avail_h / HH)
    DW, DH = int(HW * scale), int(HH * scale)
    OX, OY = (W - DW) // 2, STAGE_TOP + (avail_h - DH) // 2
    host_disp = host.resize((DW, DH), Image.LANCZOS)
    before_disp = composed_before.resize((DW, DH), Image.LANCZOS)
    corr_disp = corrected.resize((DW, DH), Image.LANCZOS)

    mask_disp = np.array(Image.fromarray(mask).resize((DW, DH), Image.NEAREST)) > 127
    gl = np.zeros((DH, DW, 4), np.uint8); gl[mask_disp] = (*GREEN, 255)
    GREEN_RGBA = Image.fromarray(gl)

    def h2d(x, y):
        return OX + x * scale, OY + y * scale

    bx1, by1 = h2d(bbox[0], bbox[1]); bx2, by2 = h2d(bbox[2], bbox[3])
    BBOX_D = (bx1, by1, bx2, by2)

    THUMB_W = 300
    THUMB = inst_src.resize((THUMB_W, int(inst_src.size[1] * THUMB_W / inst_src.size[0])), Image.LANCZOS)
    THUMB_POS = (W - THUMB_W - 70, 470)
    HIST_W = 360; HIST_POS = (W - HIST_W - 60, 70)

    SEGS = [("intro", 2.0), ("host", 1.6), ("seg", 2.2), ("dist", 2.6), ("place", 2.0),
            ("match", 2.2), ("fly", 1.6), ("blend", 1.3), ("lshift", 2.0), ("final", 2.6), ("outro", 2.6)]
    T = {}; acc = 0.0
    for nm, d in SEGS:
        T[nm] = (acc, acc + d); acc += d
    TOTAL = acc

    CAPS = {
        "host": ("① Host 이미지", "실제 학습 장면 — 기존 GT: %s" % " · ".join(meta["gt_classes"]), BLUE),
        "seg": ("② SegFormer Scene Parsing", "ground / road 영역을 검출해 배치 가능 영역 결정", BLUE),
        "dist": ("③ Inverse-Frequency 샘플링 (T=2.0)", "희소 분포 보강 → target: %s, (cx, cy)" % tgt["cls"], ORANGE),
        "place": ("④ Scale + Placement", "depth-free scale (%s) + region·IoU → bbox" % meta["scale_method"], ORANGE),
        "match": ("⑤ Lab χ² Style-match", "색감이 맞는 인스턴스 선택 (top-8 중, χ²=%.2f)" % meta["chi2"], RED),
        "fly": ("⑥ 배치", "선택된 인스턴스를 target 위치·크기로 이동", RED),
        "blend": ("⑦ Alpha-blend", "인스턴스를 host 이미지에 합성", RED),
        "lshift": ("⑧ Post L-channel 보정", "밝기 차이 보정 (≤ 20) 으로 색온도 정합", RED),
        "final": ("⑨ 증강 완료", "새 객체 + YOLO 라벨 생성 → 학습 데이터셋 합류", PURPLE),
    }

    def seg_of(t):
        for nm, (a, b) in T.items():
            if a <= t < b:
                return nm, clamp01((t - a) / (b - a))
        return "outro", 1.0

    def draw_caption(d, name, accent):
        if name not in CAPS:
            return
        title, body, _ = CAPS[name]; by = H - CAP_H
        d.rectangle([0, by, W, H], fill=BG); d.rectangle([0, by, W, by + 5], fill=accent)
        d.text((70, by + 26), title, font=F_CAP, fill=accent)
        d.text((70, by + 94), body, font=F_SUB, fill=(232, 236, 240))

    def draw_banner(d, text, accent):
        bb = d.textbbox((0, 0), text, font=F_BANNER); bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        d.rounded_rectangle([34, 20, 34 + bw + 46, 20 + bh + 26], radius=20, fill=accent)
        d.text((57, 28), text, font=F_BANNER, fill=(255, 255, 255))

    def dashed_rect(d, box, color, width=4, dash=16, gap=10, phase=0):
        x1, y1, x2, y2 = box
        def seg(xs, ys, horiz, length):
            pos = 0
            while pos < length:
                s = min(dash, length - pos)
                if horiz:
                    d.line([(xs + pos, ys), (xs + min(pos + s, length), ys)], fill=color, width=width)
                else:
                    d.line([(xs, ys + pos), (xs, ys + min(pos + s, length))], fill=color, width=width)
                pos += dash + gap
        seg(x1, y1, True, x2 - x1); seg(x1, y2, True, x2 - x1)
        seg(x1, y1, False, y2 - y1); seg(x2, y1, False, y2 - y1)

    def paste_panel(canvas, img, pos, accent, alpha=1.0, label=None):
        x, y = pos
        if alpha < 1.0:
            base = canvas.crop((x, y, x + img.size[0], y + img.size[1])).convert("RGB")
            img = Image.blend(base, img.convert("RGB"), alpha)
        canvas.paste(img, (x, y))
        dd = ImageDraw.Draw(canvas)
        dd.rectangle([x - 3, y - 3, x + img.size[0] + 3, y + img.size[1] + 3], outline=accent, width=4)
        if label:
            dd.text((x, y - 38), label, font=F_TAG, fill=accent)

    def card(lines, accent, right_img=None):
        c = Image.new("RGB", (W, H), BG); dd = ImageDraw.Draw(c)
        dd.rectangle([0, 0, 14, H], fill=accent)
        if right_img is not None:
            rw = 720; rh = int(right_img.size[1] * rw / right_img.size[0])
            rh = min(rh, 760); rw = int(right_img.size[0] * rh / right_img.size[1])
            im = right_img.resize((rw, rh), Image.LANCZOS); ox, oy = W - rw - 120, (H - rh) // 2
            c.paste(im, (ox, oy)); dd.rectangle([ox - 4, oy - 4, ox + rw + 4, oy + rh + 4], outline=accent, width=5)
        y = 320
        for ln, fnt, fill in lines:
            dd.text((130, y), ln, font=fnt, fill=fill); bb = dd.textbbox((0, 0), ln, font=fnt)
            y += (bb[3] - bb[1]) + 34
        return c

    def base_image(t):
        be = T["blend"][1]; la, lb = T["lshift"]
        if t >= lb:
            return corr_disp.copy()
        if t >= be:
            if t >= la:
                return Image.blend(before_disp, corr_disp, ease(clamp01((t - la) / (lb - la))))
            return before_disp.copy()
        return host_disp.copy()

    def render(t):
        name, p = seg_of(t)
        if name == "intro":
            intro = card([("한 장의 이미지가 증강되기까지", F_TITLE, (255, 255, 255)),
                          ("Distribution-aware Generative Copy-Paste", F_SUB, (150, 195, 255)),
                          ("", F_SUB, BG), ("실제 파이프라인을 한 예시로 따라갑니다", F_SUB, (210, 214, 220))], BLUE)
            a = ease(clamp01(p / 0.4)) if p < 0.4 else 1.0
            return Image.blend(Image.new("RGB", (W, H), BG), intro, a)
        if name == "outro":
            outro = card([("증강 완료", F_TITLE, (255, 255, 255)), ("", F_SUB, BG),
                          ("· SegFormer 영역 + inverse-frequency 분포 보정", F_SUB, (210, 214, 220)),
                          ("· Lab χ² 스타일 매칭 + L-channel 보정", F_SUB, (210, 214, 220)),
                          ("", F_SUB, BG), ("→ 장면 맥락에 맞는 자연스러운 합성 데이터", F_SUB, (255, 200, 120))],
                         PURPLE, right_img=corrected)
            a = ease(clamp01(p / 0.35)) if p < 0.35 else (1.0 if p < 0.85 else 1 - ease((p - 0.85) / 0.15))
            return Image.blend(Image.new("RGB", (W, H), BG), outro, a)

        canvas = Image.new("RGB", (W, H), BG)
        canvas.paste(base_image(t), (OX, OY))
        d = ImageDraw.Draw(canvas, "RGBA"); accent = CAPS[name][2]

        region_a = 0.0
        if t >= T["seg"][0]:
            region_a = (0.42 * ease(p) if p < 0.6 else 0.42 - 0.20 * ease((p - 0.6) / 0.4)) if name == "seg" else 0.18
        if region_a > 0.001:
            ov = GREEN_RGBA.copy(); ov.putalpha(GREEN_RGBA.split()[3].point(lambda a: int(a * region_a)))
            canvas.paste(ov, (OX, OY), ov)

        if name in ("host", "final") or t >= T["final"][0]:
            for gb in meta["gt_boxes"]:
                gx1, gy1 = h2d(gb[0], gb[1]); gx2, gy2 = h2d(gb[2], gb[3])
                d.rectangle([gx1, gy1, gx2, gy2], outline=(255, 255, 255, 150), width=2)

        if t >= T["dist"][0]:
            him = hist_target if t >= T["dist"][0] + (T["dist"][1] - T["dist"][0]) * 0.5 else hist_inset
            him = him.resize((HIST_W, int(him.size[1] * HIST_W / him.size[0])), Image.LANCZOS)
            ia = ease(p / 0.3) if (name == "dist" and p < 0.3) else 1.0
            paste_panel(canvas, him, HIST_POS, ORANGE, alpha=ia, label="4D joint histogram")
            dotx, doty = h2d(tgt["cx"] * HW, tgt["cy"] * HH); pr = 14 + 6 * math.sin(t * 6)
            d.ellipse([dotx - pr, doty - pr, dotx + pr, doty + pr], outline=ORANGE + (255,), width=4)

        if T["place"][0] <= t < T["blend"][1]:
            ph = int(t * 60)
            if name == "place":
                g = ease(p); cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                gb = (cx + (bx1 - cx) * g, cy + (by1 - cy) * g, cx + (bx2 - cx) * g, cy + (by2 - cy) * g)
                dashed_rect(d, gb, ORANGE + (255,), phase=ph)
            else:
                dashed_rect(d, BBOX_D, ORANGE + (255,), phase=ph)

        if T["match"][0] <= t < T["fly"][0] or name == "match":
            ia = ease(clamp01((t - T["match"][0]) / 0.5))
            paste_panel(canvas, THUMB.convert("RGB"), THUMB_POS, RED, alpha=ia, label="selected instance")

        if name == "fly":
            g = ease(p); sx0, sy0 = THUMB_POS; sw0, sh0 = THUMB.size
            cx0, cy0 = sx0 + sw0 / 2, sy0 + sh0 / 2; cx1, cy1 = (bx1 + bx2) / 2, (by1 + by2) / 2
            cx = cx0 + (cx1 - cx0) * g; cy = cy0 + (cy1 - cy0) * g
            cw = sw0 + ((bx2 - bx1) - sw0) * g; ch = sh0 + ((by2 - by1) - sh0) * g
            fim = inst_placed.resize((max(2, int(cw)), max(2, int(ch))), Image.LANCZOS)
            canvas.paste(fim, (int(cx - cw / 2), int(cy - ch / 2)), fim)
        elif name == "blend":
            fim = inst_placed.resize((int(bx2 - bx1), int(by2 - by1)), Image.LANCZOS)
            canvas.paste(fim, (int(bx1), int(by1)), fim)

        if name == "final":
            d.rectangle(BBOX_D, outline=PURPLE + (255,), width=5)
            lbl = tgt["cls"]; lb = d.textbbox((0, 0), lbl, font=F_TAG); lw, lh = lb[2] - lb[0], lb[3] - lb[1]
            d.rectangle([bx1, by1 - lh - 14, bx1 + lw + 16, by1], fill=PURPLE + (255,))
            d.text((bx1 + 8, by1 - lh - 12), lbl, font=F_TAG, fill=(255, 255, 255))
            if p > 0.3:
                sa = ease(clamp01((p - 0.3) / 0.4)); stamp = "Augmented ✓"
                sb = d.textbbox((0, 0), stamp, font=F_CAP); sw = sb[2] - sb[0]
                sx = OX + DW - sw - 40; sy = OY + 24
                d.rounded_rectangle([sx - 18, sy - 8, sx + sw + 18, sy + 60], radius=14, fill=(15, 170, 90, int(230 * sa)))
                d.text((sx, sy), stamp, font=F_CAP, fill=(255, 255, 255, int(255 * sa)))

        draw_banner(d, "REAL PIPELINE · " + meta["host_file"][:22], accent)
        draw_caption(d, name, accent)
        return canvas

    frame_dir = Path(tempfile.mkdtemp(prefix="ptour_"))
    n = int(TOTAL * FPS)
    for i in range(n):
        render(i / FPS).save(frame_dir / ("%05d.png" % i))
    print(f"rendered {n} frames ({TOTAL:.1f}s) -> encoding")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(frame_dir / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(args.out)], check=True)
    shutil.rmtree(frame_dir, ignore_errors=True)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
