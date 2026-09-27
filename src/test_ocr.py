"""Тест Tesseract OCR на синтетическом номере."""
from pathlib import Path
from PIL import Image
import pytesseract

# Путь к Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ROOT = Path(__file__).resolve().parent.parent
img_path = ROOT / "dataset" / "images" / "synthetic" / "synth_000000.jpg"

print(f"Картинка: {img_path}")
print(f"Существует: {img_path.exists()}")
print()

img = Image.open(img_path)
print(f"Размер: {img.size}")

# Tesseract: только буквы A-Z + цифры
# Конфиг:
#   --psm 7 = single line of text
#   -c tessedit_char_whitelist=... = ограничение алфавита
config = "--psm 7 -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789"
text = pytesseract.image_to_string(img, config=config, lang="eng")

print(f"Распознанный текст: '{text.strip()}'")
print()

# Для сравнения — без whitelist
config2 = "--psm 7"
text2 = pytesseract.image_to_string(img, config=config2, lang="eng")
print(f"Без whitelist: '{text2.strip()}'")