"""Реалистичный генератор v2: номера на РЕАЛЬНЫХ фонах.

Отличия от v1:
- фон берётся из dataset/generator/backgrounds/ (реальные фото)
- маленький номер (5-20% кадра)
- перспективное искажение
- шум, размытие, освещение

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

BG_DIR = ROOT / "dataset" / "generator" / "backgrounds"

TEMPLATES = {
    "type1":  {"size": (520, 112), "bg": (245, 245, 245), "fg": (15, 15, 15), "font_size": 95},
    "type1a": {"size": (290, 170), "bg": (245, 245, 245), "fg": (15, 15, 15), "font_size": 75},
    "type1b": {"size": (520, 112), "bg": (250, 210, 10), "fg": (15, 15, 15), "font_size": 95},
}


def render_plate(text, plate_type, font_path):
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
    draw.rectangle([0, 0, w - 1, h - 1], outline=(0, 0, 0), width=2)
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_backgrounds(rng, target_w, target_h, bg_files):
    """Загружает случайный фон, ресайзит/кропит под (target_w, target_h)."""
    path = rng.choice(bg_files)
    bg = cv2.imread(str(path))
    if bg is None:
        # fallback — серый фон
        return np.full((target_h, target_w, 3), 128, dtype=np.uint8)

    bh, bw = bg.shape[:2]
    # Масштабируем так, чтобы покрыть target
    scale = max(target_w / bw, target_h / bh)
    new_w = int(bw * scale)
    new_h = int(bh * scale)
    resized = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Случайный кроп
    x = rng.randint(0, max(0, new_w - target_w))
    y = rng.randint(0, max(0, new_h - target_h))
    crop = resized[y:y + target_h, x:x + target_w]
    return crop.copy()


def order_quad_clockwise_from_tl(pts):
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


def place_plate(bg, plate_img, rng):
    bg_h, bg_w = bg.shape[:2]
    ph, pw = plate_img.shape[:2]

    # Целевая ширина — 5..20% ширины фона
    target_w_frac = rng.uniform(0.05, 0.20)
    target_w = int(bg_w * target_w_frac)
    target_h = int(ph * (target_w / pw))

    plate_resized = cv2.resize(plate_img, (target_w, target_h))

    x = rng.randint(0, max(0, bg_w - target_w))
    y = rng.randint(0, max(0, bg_h - target_h))

    src_pts = np.float32([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1],
    ])
    src_pts[:, 0] += x
    src_pts[:, 1] += y

    jitter_x = target_w * rng.uniform(0.0, 0.12)
    jitter_y = target_h * rng.uniform(0.0, 0.12)

    def jitter(pt):
        return [
            pt[0] + rng.uniform(-jitter_x, jitter_x),
            pt[1] + rng.uniform(-jitter_y, jitter_y),
        ]

    dst_pts = np.float32([jitter(p) for p in src_pts])
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(
        plate_resized, H, (bg_w, bg_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    src_mask = np.full((target_h, target_w), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        src_mask, H, (bg_w, bg_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    mask_3 = (mask[..., None] / 255.0)
    result = (bg.astype(np.float32) * (1 - mask_3) + warped.astype(np.float32) * mask_3)
    result = np.clip(result, 0, 255).astype(np.uint8)

    dst_pts_ordered = order_quad_clockwise_from_tl(dst_pts)
    xs = dst_pts_ordered[:, 0]
    ys = dst_pts_ordered[:, 1]
    x1 = max(0, int(xs.min())); y1 = max(0, int(ys.min()))
    x2 = min(bg_w, int(xs.max())); y2 = min(bg_h, int(ys.max()))
    bbox = (x1, y1, x2 - x1, y2 - y1)
    return result, dst_pts_ordered, bbox


def postprocess(img, rng):
    # Размытие
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Яркость/контраст
    alpha = rng.uniform(0.7, 1.3)
    beta = rng.uniform(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Шум
    if rng.random() < 0.5:
        sigma = rng.uniform(2, 12)
        noise = np.random.default_rng(rng.randint(0, 10**9)).normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--font", default="dataset/generator/fonts/arialbd.ttf")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--clear-meta", action="store_true",
                        help="очистить meta.csv перед генерацией")
    args = parser.parse_args()

    font_path = Path(args.font)
    if not font_path.exists():
        print(f"Шрифт не найден: {font_path}", file=sys.stderr)
        return 2

    bg_files = list(BG_DIR.glob("*.png")) + list(BG_DIR.glob("*.jpg")) + list(BG_DIR.glob("*.jpeg"))
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
    if args.clear_meta and meta_path.exists():
        meta_path.unlink()
    write_header = not meta_path.exists()

    rows = []

    for i in range(args.n):
        bg_w = rng.randint(800, 1920)
        bg_h = rng.randint(600, 1080)

        bg = load_backgrounds(rng, bg_w, bg_h, bg_files)

        ptype = rng.choice(TARGET_TYPES)
        text = make_random_plate(ptype, rng)
        plate = render_plate(text, ptype, str(font_path))

        img, quad, bbox = place_plate(bg, plate, rng)
        img = postprocess(img, rng)

        name = f"synth_real_{i:06d}.jpg"
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

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n}")

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