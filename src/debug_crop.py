"""Вырезает crop номера из изображения и сохраняет для просмотра."""
import cv2
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
model = YOLO(str(ROOT / "runs" / "detect" / "runs" / "synth_v2" / "weights" / "best.pt"))

# Возьмём img_010 (там модель нашла 4 знака)
img_path = ROOT / "data" / "debug" / "debug set" / "img_010.jpg"
img = cv2.imread(str(img_path))

results = model(img, verbose=False, conf=0.25)
r = results[0]

out_dir = ROOT / "debug_crops"
out_dir.mkdir(exist_ok=True)

for i in range(len(r.boxes)):
    x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
    cls = int(r.boxes.cls[i].cpu().numpy())
    conf = float(r.boxes.conf[i].cpu().numpy())
    
    crop = img[y1:y2, x1:x2]
    # Сохраним оригинал
    cv2.imwrite(str(out_dir / f"crop_{i}_orig.jpg"), crop)
    # И апскейл ×5
    h, w = crop.shape[:2]
    big = cv2.resize(crop, (w*5, h*5), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(out_dir / f"crop_{i}_x5.jpg"), big)
    
    print(f"Crop {i}: class={cls}, conf={conf:.3f}, size={w}x{h}")

print(f"Сохранено в {out_dir}")