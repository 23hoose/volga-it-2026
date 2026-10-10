"""Inference v3: YOLO (детекция) + Tesseract (улучшенный препроцессинг)."""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytesseract
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from plate_utils import normalize_plate


CLASS_NAMES = {0: "type1", 1: "type1a", 2: "type1b", 3: "other"}
VALID_CHARS = "ABEKMHOPCTYX0123456789"


def trim_black_borders(img):
    """Убирает чёрные полосы по краям crop'а номера."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Порог: пиксели темнее 60 считаем «чёрной рамкой»
    mask = gray > 60
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Небольшой отступ, чтобы не отрезать буквы
    pad = 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(img.shape[0], y1 + pad)
    x1 = min(img.shape[1], x1 + pad)

    return img[y0:y1, x0:x1]


def ocr_plate(crop_bgr):
    """Улучшенное распознавание: обрезка рамки, апскейл ×5, морфология."""
    if crop_bgr is None or crop_bgr.size == 0:
        return ""

    # 1. Убираем чёрную рамку
    trimmed = trim_black_borders(crop_bgr)
    if trimmed.size == 0:
        return ""

    # 2. Апскейл ×5 (или до высоты 120)
    h, w = trimmed.shape[:2]
    target_h = max(h * 5, 120)
    scale = target_h / h
    target_w = int(w * scale)
    big = cv2.resize(trimmed, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # 3. Grayscale
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    # 4. Размытие (убрать шум)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 5. Бинаризация Оцу
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Инвертировать если текст белый на чёрном
    if np.mean(binary) < 127:
        binary = 255 - binary

    # 7. Морфология: закрыть дырки в буквах (не сильно!)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 8. Обрезать белые поля после бинаризации
    coords = np.argwhere(binary < 128)
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        pad = 5
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(binary.shape[0], y1 + pad)
        x1 = min(binary.shape[1], x1 + pad)
        binary = binary[y0:y1, x0:x1]

    pil_img = Image.fromarray(binary)

    # 9. Пробуем разные PSM, возвращаем самое длинное валидное
    best_text = ""
    for psm in [7, 8, 6]:
        config = f"--psm {psm} -c tessedit_char_whitelist={VALID_CHARS}"
        try:
            text = pytesseract.image_to_string(pil_img, config=config, lang="eng")
            text = normalize_plate(text)
            # Если длина 7-9 — считаем валидным
            if 7 <= len(text) <= 9:
                return text
            # Иначе сохраняем как лучшее
            if len(text) > len(best_text):
                best_text = text
        except Exception:
            continue

    return best_text


def main():
    parser = argparse.ArgumentParser(description="Inference v3: YOLO + Tesseract (улучшенный)")
    parser.add_argument("images", help="каталог с изображениями")
    parser.add_argument("--model", default="runs/detect/runs/finetune_v1/weights/best.pt")
    parser.add_argument("--output", "-o", default="result_v3.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"Каталог не найден: {images_dir}", file=sys.stderr)
        return 2

    model_path = ROOT / args.model
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}", file=sys.stderr)
        return 2

    print(f"Загрузка YOLO из {model_path}...")
    model = YOLO(str(model_path))

    exts = (".jpg", ".jpeg", ".png")
    files = sorted(p for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in exts)
    print(f"Изображений: {len(files)}")
    print()

    rows = []
    for i, img_path in enumerate(files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model(img, verbose=False, conf=args.conf)
        r = results[0]
        n_det = len(r.boxes) if r.boxes is not None else 0

        detections = []
        if n_det > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for j in range(n_det):
                x1, y1, x2, y2 = xyxy[j].astype(int)
                plate_type = CLASS_NAMES.get(int(cls[j]), "other")
                box_conf = float(confs[j])
                if plate_type == "other":
                    continue

                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(img.shape[1], x2)
                y2c = min(img.shape[0], y2)
                crop = img[y1c:y2c, x1c:x2c]

                text = ocr_plate(crop)
                detections.append((text, plate_type, box_conf))

        for text, plate_type, box_conf in detections:
            rows.append({
                "image": img_path.name,
                "plate_num": text if text else "########",
                "plate_type": plate_type,
                "confidence": f"{box_conf:.3f}",
            })

        if (i % 20 == 0) or (i == len(files)):
            print(f"[{i}/{len(files)}]")

    out_path = ROOT / args.output
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "plate_num", "plate_type", "confidence"],
                           delimiter=";", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print()
    print(f"Сохранено: {out_path}")
    print(f"Всего строк: {len(rows)}")


if __name__ == "__main__":
    main()