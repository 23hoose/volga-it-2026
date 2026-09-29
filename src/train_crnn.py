"""Обучение CRNN для распознавания текста ГРЗ.

Данные: dataset_crnn/images/*.jpg + dataset_crnn/labels.txt
Модель: CRNN (CNN + BiLSTM + CTC)
"""

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from crnn_model import CRNN, ctc_decode


CHARS = "ABEKMHOPCTYX0123456789"
BLANK = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1


class PlateDataset(Dataset):
    def __init__(self, samples, img_dir, img_h=48, img_w=256, augment=False):
        self.samples = samples
        self.img_dir = Path(img_dir)
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img_bgr):
        """Ресайз с сохранением пропорций + padding (без обрезки!)."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        scale = min(self.img_h / h, self.img_w / w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_w = self.img_w - new_w
        pad_h = self.img_h - new_h
        if pad_w < 0 or pad_h < 0:
            resized = resized[:self.img_h, :self.img_w]
            pad_w = self.img_w - resized.shape[1]
            pad_h = self.img_h - resized.shape[0]

        padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_CONSTANT, value=255)

        img = padded.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img[None, :, :]
        return img

    def __getitem__(self, idx):
        name, text = self.samples[idx]
        path = self.img_dir / name
        img = cv2.imread(str(path))
        if img is None:
            img = np.full((32, 128, 3), 255, dtype=np.uint8)

        if self.augment:
            if random.random() < 0.3:
                img = np.clip(img.astype(np.float32) * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
            if random.random() < 0.15:
                img = cv2.GaussianBlur(img, (3, 3), 0)

        img_tensor = self.preprocess(img)
        target = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
        target = torch.tensor(target, dtype=torch.long)
        return torch.from_numpy(img_tensor), target, text


def collate_fn(batch):
    imgs, targets, texts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_cat = torch.cat(targets) if targets else torch.tensor([], dtype=torch.long)
    return imgs, targets_cat, target_lengths, texts


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (imgs, targets_cat, target_lengths, texts) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            imgs = imgs.to(device)
            log_probs = model(imgs)
            decoded = ctc_decode(log_probs, blank=BLANK)

            for idx_seq, gt_text in zip(decoded, texts):
                pred_text = "".join(IDX_TO_CHAR.get(i, "?") for i in idx_seq)
                if pred_text == gt_text:
                    correct += 1
                total += 1

    acc = correct / total if total else 0.0
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset_crnn")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--img-h", type=int, default=48)
    parser.add_argument("--img-w", type=int, default=256)
    parser.add_argument("--out", default="models/crnn_v4_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Image size: {args.img_h}x{args.img_w}")

    data_dir = ROOT / args.data
    labels_path = data_dir / "labels.txt"
    img_dir = data_dir / "images"

    samples = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            samples.append((parts[0], parts[1]))

    print(f"Всего примеров: {len(samples)}")

    random.shuffle(samples)
    n_val = int(len(samples) * args.val_split)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = PlateDataset(train_samples, img_dir,
                            img_h=args.img_h, img_w=args.img_w, augment=True)
    val_ds = PlateDataset(val_samples, img_dir,
                          img_h=args.img_h, img_w=args.img_w, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    model = CRNN(num_classes=NUM_CLASSES, hidden_size=256).to(device)
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

    ctc_loss = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_acc = 0.0
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        for imgs, targets_cat, target_lengths, _texts in train_loader:
            imgs = imgs.to(device)
            targets_cat = targets_cat.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(imgs)
            T, B, C = log_probs.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, targets_cat, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"Эпоха {epoch:2d}/{args.epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "chars": CHARS,
                "epoch": epoch,
                "val_acc": val_acc,
                "img_h": args.img_h,
                "img_w": args.img_w,
            }, str(out_path))
            print(f"   + Лучшая модель: val_acc={val_acc:.4f}")

    print()
    print(f"Готово. Лучшая точность: {best_acc:.4f}")
    print(f"Модель: {out_path}")


if __name__ == "__main__":
    main()