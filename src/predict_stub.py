"""Заглушка inference: не распознаёт ничего, пишет пустой CSV.

Используется, чтобы проверить всю цепочку:
  inference → result.csv → evaluate.py → Score

Формат вывода (раздел 5 задания):
  image;plate_num;plate_type;confidence
"""

import argparse
import csv
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Заглушка inference для проверки пайплайна."
    )
    parser.add_argument(
        "images",
        help="каталог с изображениями (jpg/png)"
    )
    parser.add_argument(
        "--output", "-o",
        default="result.csv",
        help="куда сохранить CSV с результатом (по умолчанию result.csv)"
    )
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"Каталог не найден: {images_dir}", file=sys.stderr)
        return 2

    # Считаем, сколько изображений в каталоге (для отчёта)
    exts = (".jpg", ".jpeg", ".png")
    files = sorted(p for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in exts)

    print(f"Найдено изображений: {len(files)}")

    # Пишем CSV с заголовком и без строк данных
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(["image", "plate_num", "plate_type", "confidence"])

    print(f"Сохранено: {out_path.resolve()}")
    print("Строк в CSV: 0 (заглушка)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
