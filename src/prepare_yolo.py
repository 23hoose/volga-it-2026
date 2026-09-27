"""Подготовка YOLO-датасета: разделение на train/val, создание data.yaml."""

import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_IMAGES = ROOT / "dataset" / "images" / "synthetic"
SRC_LABELS = ROOT / "dataset" / "labels"
DST = ROOT / "dataset_yolo"

SEED = 42
VAL_RATIO = 0.1


def main():
    rng = random.Random(SEED)

    # Собираем все изображения
    images = sorted(SRC_IMAGES.glob("*.jpg"))
    print(f"Найдено изображений: {len(images)}")

    # Перемешиваем
    rng.shuffle(images)

    n_val = int(len(images) * VAL_RATIO)
    val_images = images[:n_val]
    train_images = images[n_val:]
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    # Создаём структуру
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DST / sub).mkdir(parents=True, exist_ok=True)

    # Копируем
    for img in train_images:
        shutil.copy(img, DST / "images" / "train" / img.name)
        lbl = SRC_LABELS / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, DST / "labels" / "train" / lbl.name)

    for img in val_images:
        shutil.copy(img, DST / "images" / "val" / img.name)
        lbl = SRC_LABELS / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, DST / "labels" / "val" / lbl.name)

    # data.yaml
    yaml_content = (
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
    (DST / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"Готово: {DST}")
    print(f"  Train: {len(train_images)}")
    print(f"  Val: {len(val_images)}")


if __name__ == "__main__":
    main()