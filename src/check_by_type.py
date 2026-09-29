"""Проверка точности CRNN по длинам номеров."""
import random
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


def preprocess(img_bgr, img_h=48, img_w=256):
    """Ресайз с сохранением пропорций + padding (без обрезки!)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    scale = min(img_h / h, img_w / w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = img_w - new_w
    pad_h = img_h - new_h
    if pad_w < 0 or pad_h < 0:
        resized = resized[:img_h, :img_w]
        pad_w = img_w - resized.shape[1]
        pad_h = img_h - resized.shape[0]

    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w,
                                 cv2.BORDER_CONSTANT, value=255)

    img = padded.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img[None, :, :]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ROOT / "models" / "crnn_best.pt", map_location=device, weights_only=False)
    model = CRNN(num_classes=ckpt["num_classes"], hidden_size=256).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    img_h = ckpt.get("img_h", 48)
    img_w = ckpt.get("img_w", 256)
    print(f"Модель: img_h={img_h}, img_w={img_w}, val_acc={ckpt.get('val_acc')}")

    labels_path = ROOT / "dataset_crnn" / "labels.txt"
    img_dir = ROOT / "dataset_crnn" / "images"

    samples = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                samples.append((parts[0], parts[1]))

    rng = random.Random(42)
    rng.shuffle(samples)
    val = samples[:5000]

    by_len = {}
    for name, gt in val:
        L = len(gt)
        if L not in by_len:
            by_len[L] = [0, 0]

    for i, (name, gt) in enumerate(val):
        img = cv2.imread(str(img_dir / name))
        if img is None:
            continue
        x = preprocess(img, img_h, img_w)
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model(x)
            decoded = ctc_decode(log_probs, blank=0)
            pred = "".join(IDX_TO_CHAR.get(i, "?") for i in decoded[0])
        by_len[len(gt)][1] += 1
        if pred == gt:
            by_len[len(gt)][0] += 1

    print()
    print(f"{'длина':>6} {'верно':>8} {'всего':>8} {'точность':>10}")
    for L in sorted(by_len.keys()):
        c, t = by_len[L]
        print(f"{L:>6} {c:>8} {t:>8} {c/t:>10.3f}")


if __name__ == "__main__":
    main()