"""Диагностика: смотрим, что CRNN предсказывает на валидации."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from crnn_model import CRNN, ctc_decode


CHARS = "ABEKMHOPCTYX0123456789"
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}


def preprocess(img_bgr, img_h=32, img_w=128):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    new_w = int(w * (img_h / h))
    new_w = max(new_w, 8)
    resized = cv2.resize(gray, (new_w, img_h), interpolation=cv2.INTER_AREA)
    if new_w > img_w:
        start = (new_w - img_w) // 2
        resized = resized[:, start:start + img_w]
    elif new_w < img_w:
        pad = img_w - new_w
        resized = cv2.copyMakeBorder(resized, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=255)
    img = resized.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img[None, :, :]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем модель
    ckpt = torch.load(ROOT / "models" / "crnn_best.pt", map_location=device, weights_only=False)
    model = CRNN(num_classes=ckpt["num_classes"], hidden_size=256).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Модель загружена (val_acc={ckpt['val_acc']:.3f})")

    # Берём несколько примеров
    labels_path = ROOT / "dataset_crnn" / "labels.txt"
    img_dir = ROOT / "dataset_crnn" / "images"

    samples = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                samples.append((parts[0], parts[1]))

    # Смотрим 20 примеров
    print("\nПервые 20:")
    correct = 0
    for name, gt in samples[:20]:
        img = cv2.imread(str(img_dir / name))
        if img is None:
            continue
        x = preprocess(img)
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model(x)
            decoded = ctc_decode(log_probs, blank=0)
            pred = "".join(IDX_TO_CHAR.get(i, "?") for i in decoded[0])
        mark = "✓" if pred == gt else "✗"
        if pred == gt:
            correct += 1
        print(f"  {mark} GT={gt:12s}  pred={pred}")

    print(f"\nПравильно: {correct}/20")

    # Оценка на всём val-наборе
    print("\nПолная оценка на 500 примерах...")
    random_state = 42
    import random
    random.seed(random_state)
    random.shuffle(samples)
    val = samples[:500]

    correct = 0
    char_correct = 0
    char_total = 0
    for i, (name, gt) in enumerate(val):
        img = cv2.imread(str(img_dir / name))
        if img is None:
            continue
        x = preprocess(img)
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model(x)
            decoded = ctc_decode(log_probs, blank=0)
            pred = "".join(IDX_TO_CHAR.get(i, "?") for i in decoded[0])
        if pred == gt:
            correct += 1
        # Посимвольная точность
        n = min(len(pred), len(gt))
        for j in range(n):
            if pred[j] == gt[j]:
                char_correct += 1
        char_total += max(len(pred), len(gt))

    print(f"Полное совпадение: {correct}/{len(val)} = {correct / len(val):.3f}")
    print(f"Посимвольная точность: {char_correct}/{char_total} = {char_correct / char_total:.3f}")


if __name__ == "__main__":
    main()