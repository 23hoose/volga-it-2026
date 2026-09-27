"""Генератор v5 — максимально простой и надёжный.

- фон = кадр целиком (без ресайза, если не слишком большой)
- номер — прямоугольник без перспективы
- номер в случайном месте с отступом
- обводка вокруг номера для контраста
"""

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from plate_utils import make_random_plate, TARGET_TYPES, CLASS_TO_ID

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEED = 42
BG_DIR = ROOT / "dataset" / "generator" / "backgrounds"

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
    """Загружает фон. Если слишком большой — уменьшает до max. Не кропит!"""
    bg = cv2.imread(str(path))
    if bg is None:
        return None
    h, w = bg.shape[:2]
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        bg = cv2.resize(bg, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bg


def paste_plate(bg, plate, x, y):
    """Просто копирует номер в фон в позицию (x, y). Без перспективы."""
    ph, pw = plate.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    # Обрезаем, если вдруг вылезли за границы
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


def add_shadow_border(plate):
    """Добавляет к номеру чёрную тень вокруг."""
    h, w = plate.shape[:2]
    border = 4
    canvas = np.zeros((h + 2 * border, w + 2 * border, 3), dtype=np.uint8)
    canvas[border:border + h, border:border + w] = plate
    return canvas, border


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--font", default="dataset/generator/fonts/arialbd.ttf")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    font_path = Path(args.font)
    if not font_path.exists():
        print(f"Шрифт не найден: {font_path}", file=sys.stderr)
        return 2

    bg_files = sorted(BG_DIR.glob("*.png")) + sorted(BG_DIR.glob("*.jpg")) + sorted(BG_DIR.glob("*.jpeg"))
    if not bg_files:
        print(f"Нет фонов в {BG_DIR}", file=sys.stderr)
        return 2
    print(f"Найдено фонов: {len(bg_files)}")

    rng = random.Random(args.seed)

    out_dir = ROOT / "dataset" / "images" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = ROOT / "dataset" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    meta_path = ROOT / "dataset" / "meta.csv"
    if args.clear and meta_path.exists():
        meta_path.unlink()
    write_header = not meta_path.exists()

    rows = []
    skipped = 0

    for i in range(args.n):
        bg_path = bg_files[i % len(bg_files)]
        bg = load_bg(bg_path)
        if bg is None:
            skipped += 1
            continue

        bg_h, bg_w = bg.shape[:2]

        ptype = rng.choice(TARGET_TYPES)
        text = make_random_plate(ptype, rng)
        plate = render_plate(text, ptype, str(font_path))

        # Целевая ширина 12..25% от ширины кадра
        target_w_frac = rng.uniform(0.12, 0.25)
        target_w = int(bg_w * target_w_frac)
        ph, pw = plate.shape[:2]
        target_h = int(ph * (target_w / pw))

        if target_w + 40 > bg_w or target_h + 40 > bg_h:
            skipped += 1
            continue

        plate_resized = cv2.resize(plate, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Позиция с отступом от края
        margin = 20
        x = rng.randint(margin, bg_w - target_w - margin)
        y = rng.randint(margin, bg_h - target_h - margin)

        result = bg.copy()
        res = paste_plate(result, plate_resized, x, y)
        if res is None:
            skipped += 1
            continue

        # bbox и quad — прямоугольник без перспективы
        x1, y1, w1, h1 = res
        x2 = x1 + w1
        y2 = y1 + h1

        bbox = (x1, y1, w1, h1)
        quad = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        quad_s = ",".join(f"{px},{py}" for px, py in quad)

        name = f"synth_{i:06d}.jpg"
        rel = f"images/synthetic/{name}"
        cv2.imwrite(str(ROOT / "dataset" / rel), result)

        rows.append({
            "image": rel,
            "plate_num": text,
            "plate_type": ptype,
            "bbox": f"{x1},{y1},{w1},{h1}",
            "quad": quad_s,
            "is_vehicle": "1",
            "is_synthetic": "1",
            "source": "generated",
            "license": "CC-BY-4.0",
            "conditions": "day",
        })

        cls = CLASS_TO_ID[ptype]
        cx = (x1 + w1 / 2) / bg_w
        cy = (y1 + h1 / 2) / bg_h
        w_n = w1 / bg_w
        h_n = h1 / bg_h
        line_parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{w_n:.6f}", f"{h_n:.6f}"]
        for px, py in quad:
            line_parts.append(f"{px / bg_w:.6f}")
            line_parts.append(f"{py / bg_h:.6f}")
        line = " ".join(line_parts) + "\n"
        stem = Path(rel).stem
        (labels_dir / f"{stem}.txt").write_text(line, encoding="utf-8")

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.n} (пропущено {skipped})")

    with open(meta_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["image", "plate_num", "plate_type", "bbox", "quad",
                        "is_vehicle", "is_synthetic", "source", "license", "conditions"],
            delimiter=";", lineterminator="\n",
        )
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Сгенерировано: {len(rows)} (запрошено {args.n}, пропущено {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())