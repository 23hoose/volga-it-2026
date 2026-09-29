"""Генерация 50000 синтетических номеров СРАЗУ в формат для CRNN.

Использует:
- фоны из dataset/generator/backgrounds/
- шрифт dataset/generator/fonts/arialbd.ttf
- шаблоны номеров (type1, type1a, type1b)

Сразу сохраняет вырезанные crop'ы номеров + labels.txt.
"""
import argparse
import csv
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
    """Рендерит номер на светлом фоне. Возвращает BGR."""
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


def paste_plate(bg, plate, x, y):
    ph, pw = plate.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    x2 = min(x + pw, bg_w)
    y2 = min(y + ph, bg_h)
    x = max(0, x)
    y = max(0, y)
    if x >= x2 or y >= y2:
        return None
    pw_clip = x2 - x
    ph_clip = y2 - y
    bg[y:y2, x:x2] = plate[:ph_clip, :pw_clip]
    return (x, y, pw_clip, ph_clip)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--out", default="dataset_crnn")
    parser.add_argument("--seed", type=int, default=SEED)
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
    print(f"Цель: {args.n} изображений")
    print(f"Куда: {img_dir}")
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

        target_w = int(bg_w * rng.uniform(0.12, 0.25))
        ph, pw = plate.shape[:2]
        target_h = int(ph * (target_w / pw))

        if target_w + 40 > bg_w or target_h + 40 > bg_h:
            continue

        plate_resized = cv2.resize(plate, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        margin = 20
        x = rng.randint(margin, bg_w - target_w - margin)
        y = rng.randint(margin, bg_h - target_h - margin)

        result = bg.copy()
        res = paste_plate(result, plate_resized, x, y)
        if res is None:
            continue

        x1, y1, w1, h1 = res
        # Вырезаем crop номера
        crop = result[y1:y1 + h1, x1:x1 + w1]

        name = f"synth_{i:06d}.jpg"
        cv2.imwrite(str(img_dir / name), crop)
        labels.append((name, text))
        written += 1

        if written % 5000 == 0:
            print(f"  {written}/{args.n}")

    # labels.txt
    with open(labels_path, "w", encoding="utf-8") as f:
        for name, text in labels:
            f.write(f"{name}\t{text}\n")

    print()
    print(f"Готово: {written} изображений")
    print(f"labels: {labels_path}")


if __name__ == "__main__":
    main()