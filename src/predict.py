"""Полный пайплайн inference: YOLO (детекция) + Tesseract (OCR)."""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytesseract
from ultralytics import YOLO

# Путь к Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Импорт наших утилит
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from plate_utils import normalize_plate, check_plate, TARGET_TYPES


# Классы YOLO
CLASS_NAMES = {0: "type1", 1: "type1a", 2: "type1b", 3: "other"}


def ocr_plate(crop_bgr):
    """Распознаёт текст с номера. crop_bgr — BGR-картинка номера.
    Возвращает строку (нормализованную).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return ""

    # Апскейл ×3 (Tesseract лучше читает крупное)
    h, w = crop_bgr.shape[:2]
    target_h = max(h * 3, 60)
    scale = target_h / h
    target_w = int(w * scale)
    big = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Оттенки серого + инверсия если нужно
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    # Простая бинаризация Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Проверяем: если больше белого, значит номер светлый на тёмном — инвертируем
    # (Tesseract лучше читает чёрный текст на белом)
    if np.mean(binary) < 127:
        binary = 255 - binary

    # В PIL для Tesseract
    pil_img = Image.fromarray(binary)

    # Tesseract
    config = "--psm 7 -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789"
    text = pytesseract.image_to_string(pil_img, config=config, lang="eng")
    return normalize_plate(text)


def main():
    parser = argparse.ArgumentParser(description="Inference: YOLO + Tesseract")
    parser.add_argument("images", help="каталог с изображениями")
    parser.add_argument("--model", default="runs/detect/runs/synth_v2/weights/best.pt",
                        help="путь к YOLO-модели")
    parser.add_argument("--output", "-o", default="result.csv",
                        help="куда сохранить CSV")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="порог уверенности YOLO")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"Каталог не найден: {images_dir}", file=sys.stderr)
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}", file=sys.stderr)
        return 2

    print(f"Загрузка YOLO из {model_path}...")
    model = YOLO(str(model_path))

    # Собираем изображения
    exts = (".jpg", ".jpeg", ".png")
    files = sorted(p for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in exts)
    print(f"Изображений: {len(files)}")
    print()

    rows = []
    for i, img_path in enumerate(files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{i}/{len(files)}] {img_path.name}: не читается")
            continue

        # Детекция
        results = model(img, verbose=False, conf=args.conf)
        r = results[0]
        n_det = len(r.boxes) if r.boxes is not None else 0

        detections = []
        if n_det > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()

            for j in range(n_det):
                x1, y1, x2, y2 = xyxy[j].astype(int)
                plate_type = CLASS_NAMES.get(int(cls[j]), "other")
                box_conf = float(conf[j])

                # Если other — пропускаем (не оценивается)
                if plate_type == "other":
                    continue

                # Crop
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(img.shape[1], x2)
                y2c = min(img.shape[0], y2)
                crop = img[y1c:y2c, x1c:x2c]

                # OCR
                text = ocr_plate(crop)

                # Постобработка по маске
                # Если не проходит по маске, оставляем как есть
                # (в evaluate.py ошибки формата будут учтены как «не засчитан»)

                detections.append((text, plate_type, box_conf))

        # Запись
        for text, plate_type, box_conf in detections:
            rows.append({
                "image": img_path.name,
                "plate_num": text if text else "########",
                "plate_type": plate_type,
                "confidence": f"{box_conf:.3f}",
            })

        print(f"[{i}/{len(files)}] {img_path.name}: {n_det} детекций, {len(detections)} номеров")

    # Сохранение CSV
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "plate_num", "plate_type", "confidence"],
                           delimiter=";", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print()
    print(f"Сохранено: {out_path.resolve()}")
    print(f"Всего строк: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())