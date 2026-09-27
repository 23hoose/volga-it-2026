from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parent.parent
data_dir = root / "data" / "debug" / "debug set"

# Возьмём 5 картинок с несколькими знаками и 5 обычных
samples = ["img_001.jpg", "img_018.jpg", "img_002.jpg", "img_005.jpg",
           "img_010.jpg", "img_026.jpg", "img_031.jpg", "img_044.jpg",
           "img_050.jpg", "img_100.jpg"]

# Собираем collage: 5 колонок × 2 ряда
cols, rows = 5, 2
thumb_w, thumb_h = 400, 300
canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

for i, name in enumerate(samples):
    p = data_dir / name
    if not p.exists():
        print(f"Нет файла: {name}")
        continue
    img = cv2.imread(str(p))
    if img is None:
        print(f"Не читается: {name}")
        continue
    h, w = img.shape[:2]
    print(f"{name}: {w}x{h}")

    # Пропорционально уменьшим до thumb_w × thumb_h
    scale = min(thumb_w / w, thumb_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # Вставим в canvas по центру ячейки
    r, c = divmod(i, cols)
    y0 = r * thumb_h + (thumb_h - new_h) // 2
    x0 = c * thumb_w + (thumb_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized

    # Подпишем
    cv2.putText(canvas, name, (c * thumb_w + 5, r * thumb_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Сохраняем
out = root / "data" / "debug" / "samples_collage.jpg"
cv2.imwrite(str(out), canvas)
print()
print(f"Сохранено: {out}")
print(f"Размер коллажа: {canvas.shape[1]}x{canvas.shape[0]}")
