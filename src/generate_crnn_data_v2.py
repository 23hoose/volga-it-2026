"""Генерация 50000 номеров для CRNN — с УВЕЛИЧЕНИЕМ crop'а.

Отличие от v1: crop сразу апскейлится ×3, чтобы символы были крупными.
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from plate_utils import make_random_plate, TARGET_TYPES

BG_DIR = ROOT / "dataset" / "generator" / "backgrounds"
FONT_DIR = ROOT / "dataset" / "generator" / "fonts"
SEED = 42

TEMPLATES = {
    "type1":  {"size": (520, 112), "font_size": 90},
    "type1a": {"size": (290, 170), "font_size": 75},
    "type1b": {"size": (520, 112), "font_size": 90},
}


def render_plate(text, plate_type, font_path):
    tpl = TEMPLATES[plate_type]
    w, h = tpl["size"]
    bg_color = (250, 210, 10) if plate_type == "type1b" else (245, 245, 245)
    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, tpl["font_size"])
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (w - tw) // 2 - bbox[0]
    y = (h - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=(15, 15, 15), font=font)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(0, 0, 0), width=3)
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_bg(path, max_w=1920, max_h=1080):
    bg = cv2.imread(str(path))
    if bg is None:
        return None
    h, w = bg.shape[:2]
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        bg = cv2.resize(bg, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bg


def paste_and_crop(bg, plate, x, y):
    """Вставляет номер в фон и возвращает crop + улучшенный crop."""
    ph, pw = plate.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    x2 = min(x + pw, bg_w)
    y2 = min(y + ph, bg_h)
    x = max(0, x); y = max(0, y)
    if x >= x2 or y >= y2:
        return None, None

    pw_clip = x2 - x
    ph_clip = y2 - y

    # Вставляем в фон
    bg[y:y2, x:x2] = plate[:ph_clip, :pw_clip]

    # Вырезаем crop номера (без лишнего)
    crop = bg[y:y2, x:x2].copy()

    # Апскейлим ×3 (чтобы символы были крупнее)
    crop_big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                          interpolation=cv2.INTER_LANCZOS4)
    return crop, crop_big


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--out", default="dataset_crnn")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scale", type=int, default=3, help="коэффициент апскейла")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    bg_files = sorted(BG_DIR.glob("*.png")) + sorted(BG_DIR.glob("*.jpg")) + sorted(BG_DIR.glob("*.jpeg"))
    if not bg_files:
        print(f"Нет фонов в {BG_DIR}", file=sys.stderr)
        return 2

    font_path = FONT_DIR / "arialbd.ttf"
    if not font_path.exists():
        print(f"Нет шрифта {font_path}", file=sys.stderr)
        return 2

    out_dir = ROOT / args.out
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.txt"

    print(f"Фонов: {len(bg_files)}")
    print(f"Цель: {args.n}, апскейл ×{args.scale}")
    print()

    written = 0
    labels = []

    for i in range(args.n):
        bg_path = bg_files[i % len(bg_files)]
        bg = load_bg(bg_path)
        if bg is None:
            continue

        bg_h, bg_w = bg.shape[:2]

        ptype = rng.choice(TARGET_TYPES)
        text = make_random_plate(ptype, rng)
        plate = render_plate(text, ptype, str(font_path))

        # Целевая ширина 8-15% (номера будут мелкими на фоне)
        target_w = int(bg_w * rng.uniform(0.08, 0.15))
        ph, pw = plate.shape[:2]
        target_h = int(ph * (target_w / pw))

        if target_w + 40 > bg_w or target_h + 40 > bg_h:
            continue

        plate_resized = cv2.resize(plate, (target_w, target_h),
                                   interpolation=cv2.INTER_LANCZOS4)

        margin = 20
        x = rng.randint(margin, bg_w - target_w - margin)
        y = rng.randint(margin, bg_h - target_h - margin)

        _crop, crop_big = paste_and_crop(bg.copy(), plate_resized, x, y)
        if crop_big is None:
            continue

        name = f"synth_{i:06d}.jpg"
        cv2.imwrite(str(img_dir / name), crop_big)
        labels.append((name, text))
        written += 1

        if written % 5000 == 0:
            print(f"  {written}/{args.n}")

    with open(labels_path, "w", encoding="utf-8") as f:
        for name, text in labels:
            f.write(f"{name}\t{text}\n")

    print()
    print(f"Готово: {written} изображений")
    print(f"labels: {labels_path}")


if __name__ == "__main__":
    main()