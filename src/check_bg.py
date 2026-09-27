"""Проверка: читаются ли фоны через cv2.imread."""
from pathlib import Path
import cv2

BG_DIR = Path(__file__).resolve().parent.parent / "dataset" / "generator" / "backgrounds"

files = list(BG_DIR.glob("*"))
print(f"Всего файлов в {BG_DIR}: {len(files)}")
print()

ok = 0
fail = 0
for f in sorted(files):
    img = cv2.imread(str(f))
    if img is None:
        print(f"[ОШИБКА] {f.name}: cv2.imread вернул None")
        fail += 1
    else:
        h, w = img.shape[:2]
        mean = img.mean()
        print(f"[OK]     {f.name}: {w}x{h} px, средняя яркость {mean:.1f}")
        ok += 1

print()
print(f"Загружено: {ok}, не загружено: {fail}")