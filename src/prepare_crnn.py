"""Подготовка данных для обучения CRNN: вырезает номера из синтетики."""
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "dataset" / "meta.csv"
SRC_IMG = ROOT / "dataset" / "images" / "synthetic"
DST = ROOT / "dataset_crnn"
DST_IMAGES = DST / "images"
DST_LABELS = DST / "labels.txt"

# Допустимые символы (те, что есть в ГРЗ РФ)
CHARS = "ABEKMHOPCTYX0123456789"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = blank (для CTC)


def main():
    DST_IMAGES.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(META, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            rows.append(row)

    print(f"Всего строк в meta.csv: {len(rows)}")

    written = 0
    skipped = 0
    labels = []

    for row in rows:
        img_rel = row["image"]
        img_path = ROOT / "dataset" / img_rel
        if not img_path.exists():
            skipped += 1
            continue

        # bbox = "x,y,w,h"
        try:
            x, y, w, h = map(int, row["bbox"].split(","))
        except Exception:
            skipped += 1
            continue

        if w < 5 or h < 5:
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        # Вырезаем с небольшим отступом
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        crop = img[y1:y2, x1:x2]

        # Сохраняем
        stem = Path(img_rel).stem
        out_path = DST_IMAGES / f"{stem}.jpg"
        cv2.imwrite(str(out_path), crop)

        # Текст
        text = row["plate_num"].upper().replace("#", "")
        if not text:
            skipped += 1
            continue

        labels.append((stem + ".jpg", text))
        written += 1

        if written % 1000 == 0:
            print(f"  {written}...")

    # Записываем labels.txt
    with open(DST_LABELS, "w", encoding="utf-8") as f:
        for name, text in labels:
            f.write(f"{name}\t{text}\n")

    print(f"\nГотово.")
    print(f"  Сохранено: {written}")
    print(f"  Пропущено: {skipped}")
    print(f"  Картинки: {DST_IMAGES}")
    print(f"  Разметка: {DST_LABELS}")

    # Статистика по типам
    lens = {}
    for _, text in labels:
        lens[len(text)] = lens.get(len(text), 0) + 1
    print(f"  Длина текстов: {sorted(lens.items())}")


if __name__ == "__main__":
    main()