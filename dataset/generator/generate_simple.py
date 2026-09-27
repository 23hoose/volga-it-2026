import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from plate_utils import make_random_plate, TARGET_TYPES, CLASS_TO_ID

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SEED = 42

TEMPLATES = {
    "type1":  {"size": (520, 112), "bg": (255, 255, 255), "fg": (0, 0, 0), "font_size": 90},
    "type1a": {"size": (290, 170), "bg": (255, 255, 255), "fg": (0, 0, 0), "font_size": 70},
    "type1b": {"size": (520, 112), "bg": (255, 215, 0),   "fg": (0, 0, 0), "font_size": 90},
}


def render_plate(text, plate_type, font_path):
    tpl = TEMPLATES[plate_type]
    w, h = tpl["size"]
    img = Image.new("RGB", (w, h), tpl["bg"])
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, tpl["font_size"])
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (w - tw) // 2 - bbox[0]
    y = (h - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=tpl["fg"], font=font)
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--font", default="dataset/generator/fonts/arialbd.ttf")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    font_path = Path(args.font)
    if not font_path.exists():
        print("Шрифт не найден: " + str(font_path), file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rows = []

    out_dir = ROOT / "dataset" / "images" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        ptype = rng.choice(TARGET_TYPES)
        text = make_random_plate(ptype, rng)
        img = render_plate(text, ptype, str(font_path))
        name = "synth_simple_" + str(i).zfill(6) + ".jpg"
        rel = "images/synthetic/" + name
        cv2.imwrite(str(ROOT / "dataset" / rel), img)
        h, w = img.shape[:2]
        rows.append({
            "image": rel,
            "plate_num": text,
            "plate_type": ptype,
            "bbox": "0,0," + str(w) + "," + str(h),
            "quad": "0,0," + str(w) + ",0," + str(w) + "," + str(h) + ",0," + str(h),
            "is_vehicle": "1",
            "is_synthetic": "1",
            "source": "generated",
            "license": "CC-BY-4.0",
            "conditions": "day",
        })

    meta_path = ROOT / "dataset" / "meta.csv"
    write_header = not meta_path.exists()
    with open(meta_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["image", "plate_num", "plate_type", "bbox", "quad",
                        "is_vehicle", "is_synthetic", "source", "license", "conditions"],
            delimiter=";",
            lineterminator="\n",
        )
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)

    labels_dir = ROOT / "dataset" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        stem = Path(row["image"]).stem
        cls = CLASS_TO_ID[row["plate_type"]]
        content = str(cls) + " 0.5 0.5 1.0 1.0\n"
        (labels_dir / (stem + ".txt")).write_text(content, encoding="utf-8")

    print("Сгенерировано: " + str(args.n))
    print("meta.csv: " + str(meta_path))
    print("labels: " + str(labels_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())