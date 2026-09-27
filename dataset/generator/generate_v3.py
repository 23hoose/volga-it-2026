"""Генератор синтетики v3 — финальный.

Ключевые принципы:
- номер всегда в границах кадра с отступом
- номер 15-35% ширины кадра (YOLO должен видеть)
- перспектива ограничена
- белая обводка + тень + лёгкое свечение вокруг номера
- фон ресайзится с сохранением качества
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
    """Рендерит номер с белым фоном, чёрным текстом, чёрной рамкой.
    Дополнительно — белая обводка вокруг, чтобы номер выделялся на тёмном фоне.
    """
    tpl = TEMPLATES[plate_type]
    w, h = tpl["size"]

    # Основной номер
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    if plate_type == "type1b":
        # Жёлтый фон для такси
        img.paste((250, 210, 10), (0, 0, w, h))

    font = ImageFont.truetype(font_path, tpl["font_size"])
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


def add_border_and_shadow(plate):
    """Добавляет к номеру белую обводку снаружи.
    Возвращает картинку с обводкой и альфа-маску."""
    h, w = plate.shape[:2]
    border = 6  # пикселей обводки

    # Создаём canvas с белой обводкой
    out = np.full((h + 2 * border, w + 2 * border, 3), 255, dtype=np.uint8)
    out[border:border + h, border:border + w] = plate

    # Маска: всё что не белое — считаем «номер с обводкой»
    # (белая обводка тоже часть номера для детекции, потому что она визуально слита)
    mask = np.full((h + 2 * border, w + 2 * border), 255, dtype=np.uint8)
    return out, mask


def load_background(rng, bg_files):
    """Загружает случайный фон. Возвращает numpy (BGR)."""
    path = rng.choice(bg_files)
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img


def fit_background(bg, target_w, target_h, rng):
    """Ресайзит и кропит фон, сохраняя пропорции. Высокое качество."""
    bh, bw = bg.shape[:2]

    # Соотношения сторон
    src_ratio = bw / bh
    tgt_ratio = target_w / target_h

    if src_ratio > tgt_ratio:
        # Фон шире — ресайзим по высоте, потом кропим по ширине
        new_h = target_h
        new_w = int(bw * (target_h / bh))
        interp = cv2.INTER_AREA if new_w < bw else cv2.INTER_LANCZOS4
        resized = cv2.resize(bg, (new_w, new_h), interpolation=interp)
        x = rng.randint(0, max(0, new_w - target_w))
        y = 0
    else:
        # Фон выше — ресайзим по ширине, потом кропим по высоте
        new_w = target_w
        new_h = int(bh * (target_w / bw))
        interp = cv2.INTER_AREA if new_h < bh else cv2.INTER_LANCZOS4
        resized = cv2.resize(bg, (new_w, new_h), interpolation=interp)
        x = 0
        y = rng.randint(0, max(0, new_h - target_h))

    return resized[y:y + target_h, x:x + target_w].copy()


def place_plate_with_check(bg, plate, rng):
    """Пытается вставить номер так, чтобы он был в границах кадра.
    Возвращает (img, quad, bbox) или (None, None, None) при неудаче.
    """
    bg_h, bg_w = bg.shape[:2]
    ph, pw = plate.shape[:2]

    # Целевая ширина — 15..35% ширины кадра
    target_w_frac = rng.uniform(0.15, 0.35)
    target_w = int(bg_w * target_w_frac)
    target_h = int(ph * (target_w / pw))

    # Проверка: номер не должен вылезать за кадр
    margin = 20  # пикселей отступа от края
    if target_w + 2 * margin > bg_w or target_h + 2 * margin > bg_h:
        return None, None, None

    # Позиция: с отступом от края
    x = rng.randint(margin, bg_w - target_w - margin)
    y = rng.randint(margin, bg_h - target_h - margin)

    plate_resized = cv2.resize(plate, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    src_pts = np.float32([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1],
    ])
    src_pts[:, 0] += x
    src_pts[:, 1] += y

    # Ограниченная перспектива: не более 8% от размера номера
    max_jitter_x = target_w * 0.08
    max_jitter_y = target_h * 0.08

    def jitter(pt):
        return [
            pt[0] + rng.uniform(-max_jitter_x, max_jitter_x),
            pt[1] + rng.uniform(-max_jitter_y, max_jitter_y),
        ]

    dst_pts = np.float32([jitter(p) for p in src_pts])
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(
        plate_resized, H, (bg_w, bg_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    src_mask = np.full((target_h, target_w), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        src_mask, H, (bg_w, bg_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Накладываем с лёгкой тенью вокруг (dilate маски)
    kernel = np.ones((7, 7), np.uint8)
    mask_shadow = cv2.dilate(mask, kernel, iterations=1)
    # Тень = область mask_shadow минус mask
    shadow_alpha = ((mask_shadow > 0) & (mask == 0)).astype(np.float32) * 0.5

    result = bg.astype(np.float32).copy()
    # Накладываем тень
    for c in range(3):
        result[..., c] = result[..., c] * (1 - shadow_alpha) + 0 * shadow_alpha
    # Накладываем номер
    mask_3 = (mask[..., None] / 255.0)
    result = result * (1 - mask_3) + warped.astype(np.float32) * mask_3
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Упорядочиваем quad
    quad = order_quad(dst_pts)
    xs = quad[:, 0]
    ys = quad[:, 1]
    x1 = max(0, int(xs.min())); y1 = max(0, int(ys.min()))
    x2 = min(bg_w, int(xs.max())); y2 = min(bg_h, int(ys.max()))
    bbox = (x1, y1, x2 - x1, y2 - y1)
    return result, quad, bbox


def order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    sums = pts[:, 0] + pts[:, 1]
    tl_idx = int(np.argmin(sums))
    pts = np.roll(pts, -tl_idx, axis=0)
    s = 0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        s += x1 * y2 - x2 * y1
    if s < 0:
        pts = np.array([pts[0], pts[3], pts[2], pts[1]], dtype=np.float32)
    return pts


def mild_postprocess(img, rng):
    """Очень лёгкие искажения, чтобы номер остался видимым."""
    # Только иногда — небольшая яркость/контраст
    if rng.random() < 0.5:
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.uniform(-15, 15)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Очень редко — лёгкое размытие
    if rng.random() < 0.1:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--font", default="dataset/generator/fonts/arialbd.ttf")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--clear", action="store_true",
                        help="очистить meta.csv и labels перед генерацией")
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
    tries_per_image = 5

    for i in range(args.n):
        # Пытаемся сгенерировать изображение (может не получиться — маленький кадр)
        for attempt in range(tries_per_image):
            # Размер кадра: от 1280×720 до 1920×1080
            bg_w = rng.randint(1280, 1920)
            bg_h = rng.randint(720, 1080)

            bg = load_background(rng, bg_files)
            if bg is None:
                continue
            bg = fit_background(bg, bg_w, bg_h, rng)

            ptype = rng.choice(TARGET_TYPES)
            text = make_random_plate(ptype, rng)
            plate = render_plate(text, ptype, str(font_path))

            img, quad, bbox = place_plate_with_check(bg, plate, rng)
            if img is None:
                continue

            img = mild_postprocess(img, rng)
            break
        else:
            print(f"  Не удалось сгенерировать {i}")
            continue

        name = f"synth_{i:06d}.jpg"
        rel = f"images/synthetic/{name}"
        cv2.imwrite(str(ROOT / "dataset" / rel), img)

        bbox_s = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        quad_pts = [(int(round(p[0])), int(round(p[1]))) for p in quad]
        quad_pts = [(max(0, min(bg_w - 1, x)), max(0, min(bg_h - 1, y))) for (x, y) in quad_pts]
        quad_s = ",".join(f"{x},{y}" for x, y in quad_pts)

        rows.append({
            "image": rel,
            "plate_num": text,
            "plate_type": ptype,
            "bbox": bbox_s,
            "quad": quad_s,
            "is_vehicle": "1",
            "is_synthetic": "1",
            "source": "generated",
            "license": "CC-BY-4.0",
            "conditions": "day",
        })

        # YOLO label
        cls = CLASS_TO_ID[ptype]
        cx = (bbox[0] + bbox[2] / 2) / bg_w
        cy = (bbox[1] + bbox[3] / 2) / bg_h
        w_n = bbox[2] / bg_w
        h_n = bbox[3] / bg_h
        cx = max(0, min(1, cx)); cy = max(0, min(1, cy))
        w_n = max(0, min(1, w_n)); h_n = max(0, min(1, h_n))
        line_parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{w_n:.6f}", f"{h_n:.6f}"]
        for x, y in quad_pts:
            line_parts.append(f"{x / bg_w:.6f}")
            line_parts.append(f"{y / bg_h:.6f}")
        line = " ".join(line_parts) + "\n"
        stem = Path(rel).stem
        (labels_dir / f"{stem}.txt").write_text(line, encoding="utf-8")

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.n}")

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

    print(f"Сгенерировано: {len(rows)} (запрошено {args.n})")
    print(f"meta.csv: {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())