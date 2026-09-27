"""Генератор синтетики v4 — используем фон ЦЕЛИКОМ, ищем спокойное место под номер."""

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


def load_bg_as_frame(path, max_w=1920, max_h=1080, min_w=800, min_h=600):
    """Загружает фон и приводит к разумному размеру, СОХРАНЯЯ ВСЁ изображение.
    Возвращает numpy или None.
    """
    bg = cv2.imread(str(path))
    if bg is None:
        return None

    h, w = bg.shape[:2]

    # Если слишком большой — уменьшаем, сохраняя пропорции
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        bg = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    # Если слишком маленький — увеличиваем
    if w < min_w or h < min_h:
        scale = max(min_w / w, min_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        bg = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = new_h, new_w

    return bg


def find_quiet_spot(bg, plate_w, plate_h, rng, n_candidates=30):
    """Ищет место для номера: спокойный участок фона.
    Возвращает (x, y) — левый верхний угол номера, или None.
    """
    bg_h, bg_w = bg.shape[:2]

    # Зона поиска — внутренняя часть кадра с отступом
    margin_x = plate_w // 2
    margin_y = plate_h // 2

    if bg_w - plate_w - 2 * margin_x < 0 or bg_h - plate_h - 2 * margin_y < 0:
        # Не влезает вообще — центрируем
        if plate_w + 20 > bg_w or plate_h + 20 > bg_h:
            return None
        x = (bg_w - plate_w) // 2
        y = (bg_h - plate_h) // 2
        return x, y

    # Считаем карту яркости
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)

    candidates = []
    for _ in range(n_candidates):
        x = rng.randint(20, bg_w - plate_w - 20)
        y = rng.randint(20, bg_h - plate_h - 20)

        patch = gray[y:y + plate_h, x:x + plate_w]
        if patch.size == 0:
            continue

        mean_bright = float(patch.mean())
        std_bright = float(patch.std())

        # Нам нужен спокойный участок:
        # - средняя яркость не крайняя (номер не потеряется)
        # - низкая текстурность (std)
        # Идеальная яркость ~ 80-180
        bright_score = 1.0 - abs(mean_bright - 130) / 130  # 1.0 при 130, 0 при 0 или 260
        bright_score = max(0.0, bright_score)

        texture_score = 1.0 / (1.0 + std_bright / 30.0)  # 1.0 при std=0, ~0.5 при std=30

        score = bright_score * 0.6 + texture_score * 0.4
        candidates.append((score, x, y))

    if not candidates:
        return None

    # Сортируем по убыванию score, берём из топ-5 случайное
    candidates.sort(reverse=True)
    top = candidates[:5]
    _, x, y = rng.choice(top)
    return x, y


def place_plate(bg, plate, rng, n_attempts=10):
    """Вставляет номер в спокойное место фона."""
    bg_h, bg_w = bg.shape[:2]
    ph, pw = plate.shape[:2]

    # Целевая ширина номера — 15..30% ширины кадра
    target_w_frac = rng.uniform(0.15, 0.30)
    target_w = int(bg_w * target_w_frac)
    target_h = int(ph * (target_w / pw))

    if target_w + 40 > bg_w or target_h + 40 > bg_h:
        return None, None, None

    plate_resized = cv2.resize(plate, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    for attempt in range(n_attempts):
        spot = find_quiet_spot(bg, target_w, target_h, rng)
        if spot is None:
            continue
        x, y = spot

        src_pts = np.float32([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ])
        src_pts[:, 0] += x
        src_pts[:, 1] += y

        # Ограниченная перспектива: 0..6%
        max_jx = target_w * 0.06
        max_jy = target_h * 0.06

        def jitter(pt):
            return [
                pt[0] + rng.uniform(-max_jx, max_jx),
                pt[1] + rng.uniform(-max_jy, max_jy),
            ]

        dst_pts = np.float32([jitter(p) for p in src_pts])

        # Проверка: все точки в границах
        if (dst_pts[:, 0].min() < 5 or dst_pts[:, 0].max() > bg_w - 5 or
                dst_pts[:, 1].min() < 5 or dst_pts[:, 1].max() > bg_h - 5):
            continue

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

        # Тень вокруг номера
        kernel = np.ones((9, 9), np.uint8)
        mask_shadow = cv2.dilate(mask, kernel, iterations=1)
        shadow_alpha = ((mask_shadow > 0) & (mask == 0)).astype(np.float32) * 0.6

        result = bg.astype(np.float32).copy()
        for c in range(3):
            result[..., c] = result[..., c] * (1 - shadow_alpha) + 0 * shadow_alpha

        mask_3 = (mask[..., None] / 255.0)
        result = result * (1 - mask_3) + warped.astype(np.float32) * mask_3
        result = np.clip(result, 0, 255).astype(np.uint8)

        quad = order_quad(dst_pts)
        xs = quad[:, 0]
        ys = quad[:, 1]
        x1 = max(0, int(xs.min())); y1 = max(0, int(ys.min()))
        x2 = min(bg_w, int(xs.max())); y2 = min(bg_h, int(ys.max()))
        bbox = (x1, y1, x2 - x1, y2 - y1)
        return result, quad, bbox

    return None, None, None


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

    # Цикл по фонам — используем каждый фон несколько раз, чтобы не тратить
    for i in range(args.n):
        bg_path = bg_files[i % len(bg_files)]  # циклично, чтобы все фоны были задействованы
        bg = load_bg_as_frame(bg_path)
        if bg is None:
            skipped += 1
            continue

        bg_h, bg_w = bg.shape[:2]

        ptype = rng.choice(TARGET_TYPES)
        text = make_random_plate(ptype, rng)
        plate = render_plate(text, ptype, str(font_path))

        img, quad, bbox = place_plate(bg, plate, rng)
        if img is None:
            skipped += 1
            continue

        name = f"synth_{i:06d}.jpg"
        rel = f"images/synthetic/{name}"
        cv2.imwrite(str(ROOT / "dataset" / rel), img)

        bbox_s = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        quad_pts = [(int(round(p[0])), int(round(p[1]))) for p in quad]
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

        cls = CLASS_TO_ID[ptype]
        cx = (bbox[0] + bbox[2] / 2) / bg_w
        cy = (bbox[1] + bbox[3] / 2) / bg_h
        w_n = bbox[2] / bg_w
        h_n = bbox[3] / bg_h
        line_parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{w_n:.6f}", f"{h_n:.6f}"]
        for x, y in quad_pts:
            line_parts.append(f"{x / bg_w:.6f}")
            line_parts.append(f"{y / bg_h:.6f}")
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
    print(f"meta.csv: {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())