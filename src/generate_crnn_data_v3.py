"""Генерация 50000 номеров для CRNN — БЕЗ обрезки, подгонка шрифта.

Ключевые изменения:
- Подбор размера шрифта: длинные номера уменьшаются, короткие — крупные.
- Все номера полностью видны, ничего не обрезается.
- Лёгкие аугментации: перспектива, шум, размытие, яркость.
- Не вставляется в фон — сразу рендерится финальный crop.
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

FONT_DIR = ROOT / "dataset" / "generator" / "fonts"
SEED = 42

TEMPLATES = {
    "type1":  {"size": (520, 112), "font_size": 90},
    "type1a": {"size": (290, 170), "font_size": 75},
    "type1b": {"size": (520, 112), "font_size": 90},
}


def render_plate(text, plate_type, font_path):
    """Рендерит номер, подгоняя размер шрифта под ширину шаблона."""
    tpl = TEMPLATES[plate_type]
    w, h = tpl["size"]
    bg_color = (250, 210, 10) if plate_type == "type1b" else (245, 245, 245)
    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    # Отступы от краёв шаблона
    margin_x = 40
    margin_y = 15
    max_w = w - 2 * margin_x
    max_h = h - 2 * margin_y

    # Подбираем размер шрифта
    font_size = tpl["font_size"]
    while font_size > 20:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= max_w and th <= max_h:
            break
        font_size -= 1

    # Рендерим финальным шрифтом
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (w - tw) // 2 - bbox[0]
    y = (h - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=(15, 15, 15), font=font)

    # Чёрная рамка
    draw.rectangle([0, 0, w - 1, h - 1], outline=(0, 0, 0), width=3)

    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def add_perspective(img, rng, max_shift=0.03):
    """Лёгкая перспектива."""
    h, w = img.shape[:2]
    shift_x = int(w * max_shift)
    shift_y = int(h * max_shift)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [rng.randint(0, shift_x), rng.randint(0, shift_y)],
        [w - rng.randint(0, shift_x), rng.randint(0, shift_y)],
        [w - rng.randint(0, shift_x), h - rng.randint(0, shift_y)],
        [rng.randint(0, shift_x), h - rng.randint(0, shift_y)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def add_noise(img, rng, sigma=5):
    """Лёгкий шум."""
    noise = np.random.default_rng(rng.randint(0, 10**9)).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    font_path = FONT_DIR / "RegistrationPlateUk-P4vB.ttf"
    if not font_path.exists():
        print(f"Нет шрифта {font_path}", file=sys.stderr)
        return 2

    out_dir = ROOT / args.out
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.txt"

    print(f"Цель: {args.n} изображений")
    print()

    written = 0
    labels = []

    for i in range(args.n):
        ptype = rng.choice(TARGET_TYPES)
        text = make_random_plate(ptype, rng)
        plate = render_plate(text, ptype, str(font_path))

        # Лёгкая перспектива
        if rng.random() < 0.5:
            plate = add_perspective(plate, rng, max_shift=0.03)

        # Лёгкий шум
        if rng.random() < 0.3:
            plate = add_noise(plate, rng, sigma=rng.uniform(3, 8))

        # Лёгкое размытие
        if rng.random() < 0.2:
            k = rng.choice([3, 5])
            plate = cv2.GaussianBlur(plate, (k, k), 0)

        # Лёгкое изменение яркости
        if rng.random() < 0.3:
            alpha = rng.uniform(0.85, 1.15)
            beta = rng.uniform(-15, 15)
            plate = cv2.convertScaleAbs(plate, alpha=alpha, beta=beta)

        name = f"synth_{i:06d}.jpg"
        cv2.imwrite(str(img_dir / name), plate)
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