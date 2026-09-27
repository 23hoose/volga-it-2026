"""Подготовка данных для дообучения: смесь реальных + синтетических.

- Реальные: 100 изображений с разметкой в data/debug/labels_yolo
- Синтетика: dataset/images/synthetic + dataset/labels
- Разделение 90/10 train/val
"""
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Источники
REAL_IMG_DIR = ROOT / "data" / "debug" / "debug set"
REAL_LBL_DIR = ROOT / "data" / "debug" / "labels_yolo"
SYNTH_IMG_DIR = ROOT / "dataset" / "images" / "synthetic"
SYNTH_LBL_DIR = ROOT / "dataset" / "labels"

# Куда
DST = ROOT / "dataset_finetune"
SEED = 42
VAL_RATIO = 0.1


def copy_pairs(pairs, split, dst):
    for img_path, lbl_path in pairs:
        shutil.copy(img_path, dst / "images" / split / img_path.name)
        if lbl_path.exists():
            shutil.copy(lbl_path, dst / "labels" / split / lbl_path.name)


def main():
    rng = random.Random(SEED)

    # Структура
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DST / sub).mkdir(parents=True, exist_ok=True)

    # Реальные — с разметкой
    real_pairs = []
    for lbl in REAL_LBL_DIR.glob("*.txt"):
        img = REAL_IMG_DIR / (lbl.stem + ".jpg")
        if img.exists() and lbl.stat().st_size > 0:
            real_pairs.append((img, lbl))
    print(f"Реальных пар: {len(real_pairs)}")

    # Синтетика
    synth_pairs = []
    for img in SYNTH_IMG_DIR.glob("*.jpg"):
        lbl = SYNTH_LBL_DIR / (img.stem + ".txt")
        if lbl.exists():
            synth_pairs.append((img, lbl))
    print(f"Синтетических пар: {len(synth_pairs)}")

    # Разделяем
    rng.shuffle(real_pairs)
    rng.shuffle(synth_pairs)

    n_real_val = max(1, int(len(real_pairs) * VAL_RATIO))
    real_val = real_pairs[:n_real_val]
    real_train = real_pairs[n_real_val:]

    n_synth_val = int(len(synth_pairs) * VAL_RATIO)
    synth_val = synth_pairs[:n_synth_val]
    synth_train = synth_pairs[n_synth_val:]

    print(f"Реальные:  train={len(real_train)}, val={len(real_val)}")
    print(f"Синтетика: train={len(synth_train)}, val={len(synth_val)}")

    # Копируем
    copy_pairs(real_train, "train", DST)
    copy_pairs(synth_train, "train", DST)
    copy_pairs(real_val, "val", DST)
    copy_pairs(synth_val, "val", DST)

    # data.yaml
    yaml = (
        f"path: {DST.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: type1\n"
        "  1: type1a\n"
        "  2: type1b\n"
        "  3: other\n"
    )
    (DST / "data.yaml").write_text(yaml, encoding="utf-8")

    print()
    print(f"Готово: {DST}")
    print(f"Train: {(DST / 'images' / 'train').glob('*').__len__()}")
    print(f"Val:   {(DST / 'images' / 'val').glob('*').__len__()}")


if __name__ == "__main__":
    main()