"""Проверка: пересекаются ли train и val."""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "dataset_crnn" / "labels.txt"

samples = []
with open(LABELS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            samples.append(parts[0])

print(f"Всего примеров: {len(samples)}")

# Тот же seed, что в train_crnn.py
rng = random.Random(42)
samples_shuffled = samples.copy()
rng.shuffle(samples_shuffled)

val_split = 0.1
n_val = int(len(samples_shuffled) * val_split)
val = set(samples_shuffled[:n_val])
train = set(samples_shuffled[n_val:])

print(f"Train: {len(train)}")
print(f"Val: {len(val)}")

overlap = train & val
print(f"Пересечение train/val: {len(overlap)}")

if overlap:
    print("⚠️ ЕСТЬ ПЕРЕСЕЧЕНИЕ!")
    for name in list(overlap)[:10]:
        print(f"  {name}")
else:
    print("✅ Пересечения нет.")