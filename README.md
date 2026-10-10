# Распознавание нестандартных ГРЗ (Волга-IT 2026)

Решение для задачи распознавания нестандартных государственных
регистрационных знаков (ГРЗ) РФ типов 1, 1А и 1Б. Работает офлайн,
укладывается в 100 мс на кадр.

## Состав решения

- Детектор YOLOv8s (обучен на синтетике + дообучен на 100 реальных)
- OCR Tesseract (свободный движок)
- Постобработка по маске ГРЗ

## Установка

Требуется Python 3.11+ и GPU NVIDIA (опционально).

Установи зависимости:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install ultralytics opencv-python pandas pytesseract

Отдельно установи Tesseract-OCR:
https://github.com/UB-Mannheim/tesseract/wiki
(путь по умолчанию: C:\Program Files\Tesseract-OCR\)

## Запуск

Передай каталог с изображениями и путь для результата:

    python src/predict.py "путь/к/изображениям" --output result.csv

Пример с отладочным набором:

    python src/predict.py "data/debug/debug set" --output result.csv

На выходе — CSV с полями:
image;plate_num;plate_type;confidence

## Структура

- src/predict.py — основной скрипт inference
- src/plate_utils.py — утилиты работы с масками ГРЗ
- runs/detect/runs/finetune_v1/weights/best.pt — веса YOLO
- dataset/ — синтетический датасет (5000 изображений)
- ПОЯСНИТЕЛЬНАЯ_ЗАПИСКА.md — описание подхода

## Датасет

5000 синтетических изображений ГРЗ. Подробности — в dataset/README.md.
Прошёл валидатор организаторов: 0 ERROR, 5 WARNING (объёмы реальных данных).

## Ограничения

- Датасет только синтетический.
- OCR — Tesseract, не специализирован для ГРЗ.
- Возможны ошибки на реальных фото с нестандартным шрифтом.