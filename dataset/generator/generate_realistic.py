"""Реалистичный генератор синтетики для ГРЗ.

Отличия от generate_simple.py:
- большой фон (800x600 ... 1920x1080)
- маленький номер (5-20% площади кадра)
- перспективное искажение
- шум, размытие, градиент освещения
- bbox описывает quad

Фиксированный seed — набор воспроизводится.
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


# ---------------------------------------------------------------------------
# Шаблоны номеров
# ---------------------------------------------------------------------------

TEMPLATES = {
    "type1":  {"size": (520, 112), "bg": (245, 245, 245), "fg": (15, 15, 15), "font_size": 95},
    "type1a": {"size": (290, 170), "bg": (245, 245, 245), "fg": (15, 15, 15), "font_size": 75},
    "type1b": {"size": (520, 112), "bg": (250, 210, 10), "fg": (15, 15, 15), "font_size": 95},
}


def render_plate(text, plate_type, font_path):
    """Рендерит номер на светлом фоне, возвращает BGR-numpy + alpha-маску рамки."""
    tpl = TEMPLATES[plate_type]
    w, h = tpl["size"]
    img = Image.new("RGB", (w, h), tpl["bg"])
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, tpl["font_size"])

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (w - tw) // 2 - bbox[0]
    y = (h - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=tpl["fg"], font=font)

    # Тонкая чёрная рамка по краю
    draw.rectangle([0, 0, w - 1, h - 1], outline=(0, 0, 0), width=2)

    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Синтетический фон
# ---------------------------------------------------------------------------

def make_background(w, h, rng):
    """Создаёт синтетический фон: смесь градиента и случайных пятен."""
    # Базовый градиент между двумя цветами
    c1 = np.array([rng.randint(20, 200) for _ in range(3)], dtype=np.float32)
    c2 = np.array([rng.randint(20, 200) for _ in range(3)], dtype=np.float32)

    # Линейный градиент по случайному направлению
    xs = np.linspace(0, 1, w, dtype=np.float32)
    ys = np.linspace(0, 1, h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    angle = rng.uniform(0, 3.14159)
    t = (np.cos(angle) * gx + np.sin(angle) * gy)
    t = np.clip(t, 0, 1)
    bg = c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]

    # Добавим пятна — «текстуры»
    for _ in range(rng.randint(5, 20)):
        cx = rng.randint(0, w)
        cy = rng.randint(0, h)
        radius = rng.randint(30, 250)
        color = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
        # Круглая маска
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        alpha = rng.uniform(0.1, 0.4)
        bg[mask] = bg[mask] * (1 - alpha) + color * alpha

    # Размытие для мягкости
    bg = cv2.GaussianBlur(bg.astype(np.uint8), (0, 0), sigmaX=rng.uniform(2, 8))

    # Шум
    noise = np.random.default_rng(rng.randint(0, 10**9)).normal(0, 8, bg.shape)
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return bg


# ---------------------------------------------------------------------------
# Наложение номера на фон
# ---------------------------------------------------------------------------

def place_plate(bg, plate_img, rng):
    """Вставляет номер в фон с перспективой и возвращает:
       новое изображение, quad (4 точки), bbox.
    """
    bg_h, bg_w = bg.shape[:2]
    ph, pw = plate_img.shape[:2]

    # Целевая ширина — 5..20% ширины фона
    target_w_frac = rng.uniform(0.05, 0.20)
    target_w = int(bg_w * target_w_frac)
    # Сохраняем пропорции
    target_h = int(ph * (target_w / pw))

    # Ресайз
    plate_resized = cv2.resize(plate_img, (target_w, target_h))

    # Случайное положение (левый верхний угол)
    x = rng.randint(0, max(0, bg_w - target_w))
    y = rng.randint(0, max(0, bg_h - target_h))

    # Четыре угла (в системе координат большого кадра)
    src_pts = np.float32([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1],
    ])
    # Смещение на (x, y)
    src_pts[:, 0] += x
    src_pts[:, 1] += y

    # Перспективное искажение: сдвигаем углы на 0..15% размера
    jitter_x = target_w * rng.uniform(0.0, 0.15)
    jitter_y = target_h * rng.uniform(0.0, 0.15)

    def jitter(pt):
        return [
            pt[0] + rng.uniform(-jitter_x, jitter_x),
            pt[1] + rng.uniform(-jitter_y, jitter_y),
        ]

    dst_pts = np.float32([jitter(p) for p in src_pts])

    # Гомография src -> dst
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp номера
    warped = cv2.warpPerspective(
        plate_resized, H, (bg_w, bg_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    # Маска (где номер — 255)
    src_mask = np.full((target_h, target_w), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        src_mask, H, (bg_w, bg_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Накладываем
    mask_3 = (mask[..., None] / 255.0)
    result = (bg.astype(np.float32) * (1 - mask_3) + warped.astype(np.float32) * mask_3)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Считаем bbox = ограничивающий прямоугольник по dst_pts
    dst_pts_ordered = order_quad_clockwise_from_tl(dst_pts)
    xs = dst_pts_ordered[:, 0]
    ys = dst_pts_ordered[:, 1]
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(bg_w, x2); y2 = min(bg_h, y2)

    bbox = (x1, y1, x2 - x1, y2 - y1)
    return result, dst_pts_ordered, bbox


def order_quad_clockwise_from_tl(pts):
    """Упорядочивает 4 точки: сначала левый верхний (min x+y),
    дальше по часовой стрелке."""
    pts = np.array(pts, dtype=np.float32)
    # Индекс левого верхнего угла
    sums = pts[:, 0] + pts[:, 1]
    tl_idx = int(np.argmin(sums))
    # Переставляем так, чтобы tl был первым
    pts = np.roll(pts, -tl_idx, axis=0)

    # Проверяем ориентацию. Если по часовой — оставляем, иначе разворачиваем
    # (в экранных координатах y вниз, положительная площадь = по часовой)
    s = 0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        s += x1 * y2 - x2 * y1
    if s < 0:
        pts = np.array([pts[0], pts[3], pts[2], pts[1]], dtype=np.float32)
    return pts


# ---------------------------------------------------------------------------
# Постобработка кадра
# ---------------------------------------------------------------------------

def postprocess(img, rng):
    """Случайные искажения: размытие, яркость, контраст, шум."""
    # Лёгкое размытие (motion blur или gaussian)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Яркость/контраст
    alpha = rng.uniform(0.7, 1.3)   # контраст
    beta = rng.uniform(-30, 30)     # яркость
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Шум
    if rng.random() < 0.5:
        sigma = rng.uniform(2, 12)
        noise = np.random.default_rng(rng.randint(0, 10**9)).normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--font", default="dataset/generator/fonts/arialbd.ttf")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    font_path = Path(args.font)
    if not font_path.exists():
        print(f"Шрифт не найден: {font_path}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)

    out_dir = ROOT / "dataset" / "images" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = ROOT / "dataset" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    meta_path = ROOT / "dataset" / "meta.csv"
    write_header = not meta_path.exists()

    rows = []

    for i in range(args.n):
        # Размер фона
        bg_w = rng.randint(800, 1920)
        bg_h = rng.randint(600, 1080)

        bg = make_background(bg_w, bg_h, rng)

        ptype = rng.choice(TARGET_TYPES)
        text = make_random_plate(ptype, rng)
        plate = render_plate(text, ptype, str(font_path))

        img, quad, bbox = place_plate(bg, plate, rng)
        img = postprocess(img, rng)

        name = f"synth_real_{i:06d}.jpg"
        rel = f"images/synthetic/{name}"
        cv2.imwrite(str(ROOT / "dataset" / rel), img)

        # Формат bbox: "x,y,w,h", quad: "x1,y1,x2,y2,x3,y3,x4,y4" — округляем
        bbox_s = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        quad_pts = [(int(round(p[0])), int(round(p[1]))) for p in quad]
        # Клипуем в границы
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

        # YOLO labels
        cls = CLASS_TO_ID[ptype]
        # Нормированные cx, cy, w, h
        cx = (bbox[0] + bbox[2] / 2) / bg_w
        cy = (bbox[1] + bbox[3] / 2) / bg_h
        w_n = bbox[2] / bg_w
        h_n = bbox[3] / bg_h
        # Клипуем
        cx = max(0, min(1, cx)); cy = max(0, min(1, cy))
        w_n = max(0, min(1, w_n)); h_n = max(0, min(1, h_n))
        line_parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{w_n:.6f}", f"{h_n:.6f}"]
        # + quad (8 нормированных)
        for x, y in quad_pts:
            line_parts.append(f"{x / bg_w:.6f}")
            line_parts.append(f"{y / bg_h:.6f}")
        line = " ".join(line_parts) + "\n"
        stem = Path(rel).stem
        (labels_dir / f"{stem}.txt").write_text(line, encoding="utf-8")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n}")

    # Записываем meta.csv
    with open(meta_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["image", "plate_num", "plate_type", "bbox", "quad",
                        "is_vehicle", "is_synthetic", "source", "license", "conditions"],
            delimiter=";",
            lineterminator="\n",
        )
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Сгенерировано: {args.n}")
    print(f"meta.csv: {meta_path}")
    print(f"labels: {labels_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())