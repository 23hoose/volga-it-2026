# Датасет нестандартных ГРЗ (Волга-IT 2026)

## Описание

Синтетический датасет государственных регистрационных знаков (ГРЗ)
Российской Федерации для задачи распознавания нестандартных номеров.
Включает три типа знаков:

- type1 — обычные номера (520 x 112 мм, одна строка)
- type1a — квадратные двухстрочные (290 x 170 мм)
- type1b — жёлтые номера такси (520 x 112 мм, одна строка)

## Состав

- 5000 синтетических изображений (.jpg)
- 5000 размеченных знаков (по одному на изображение)
- Все изображения сгенерированы участником (is_synthetic = 1)

## Структура

dataset/
  images/synthetic/  - 5000 синтетических изображений
  labels/            - YOLO-разметка (по одному .txt на изображение)
  generator/         - скрипт генерации + фоны + шрифты
  meta.csv           - метаданные (одна строка на знак)
  README.md          - этот файл
  LICENSE            - CC BY 4.0

## Формат meta.csv

Разделитель — точка с запятой.
Поля: image;plate_num;plate_type;bbox;quad;is_vehicle;is_synthetic;source;license;conditions

image         - путь к изображению
plate_num     - номер (латиница, верхний регистр)
plate_type    - type1 / type1a / type1b / other
bbox          - x,y,w,h в пикселях
quad          - 4 угла по часовой стрелке
is_vehicle    - 1 (знак на ТС)
is_synthetic  - 1
source        - generated
license       - CC-BY-4.0
conditions    - day

## Формат labels/

YOLO-формат: class cx cy w h (плюс опционально quad — 8 чисел).
Классы: 0 = type1, 1 = type1a, 2 = type1b, 3 = other.
Координаты нормированы в [0, 1].

## Генерация

Скрипт: generator/generate_simple.py. Seed = 42 (воспроизводимо).

Команда:
python generator/generate_simple.py --n 5000

## Источники и лицензии

Изображения — сгенерированы участником.
Шрифт RegistrationPlateUk-P4vB.ttf — свободная лицензия (fontspace.com).
Фоны — Unsplash / Pexels (CC0 / Unsplash License).

## Лицензия датасета

CC BY 4.0. См. файл LICENSE.