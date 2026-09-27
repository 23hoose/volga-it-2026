"""Тест Tesseract на простом изображении номера (номер занимает весь кадр)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "dataset" / "generator" / "fonts" / "arialbd.ttf"

# Генерируем чистый номер
text = "A123BC777"
w, h = 520, 112
img = Image.new("RGB", (w, h), (245, 245, 245))
draw = ImageDraw.Draw(img)
font = ImageFont.truetype(str(FONT), 90)
bbox = draw.textbbox((0, 0), text, font=font)
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]
x = (w - tw) // 2 - bbox[0]
y = (h - th) // 2 - bbox[1]
draw.text((x, y), text, fill=(15, 15, 15), font=font)
draw.rectangle([0, 0, w - 1, h - 1], outline=(0, 0, 0), width=3)

# Сохраним, чтобы посмотреть
out_path = ROOT / "test_plate.jpg"
img.save(out_path)
print(f"Тестовая картинка: {out_path}")
print()

# Пробуем разные PSM (page segmentation mode)
for psm in [6, 7, 8, 13]:
    config = f"--psm {psm} -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789"
    txt = pytesseract.image_to_string(img, config=config, lang="eng").strip()
    print(f"PSM {psm:2d}: '{txt}'")

# Без whitelist
for psm in [7, 8]:
    config = f"--psm {psm}"
    txt = pytesseract.image_to_string(img, config=config, lang="eng").strip()
    print(f"PSM {psm:2d} (без whitelist): '{txt}'")

# С апскейлом
img_big = img.resize((w * 3, h * 3), Image.LANCZOS)
img_big.save(ROOT / "test_plate_3x.jpg")
for psm in [7, 8]:
    config = f"--psm {psm} -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789"
    txt = pytesseract.image_to_string(img_big, config=config, lang="eng").strip()
    print(f"PSM {psm:2d} (x3): '{txt}'")