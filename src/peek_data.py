import csv
from pathlib import Path

# Пути
root = Path(__file__).resolve().parent.parent
data_dir = root / "data" / "debug" / "debug set"
labels_path = root / "data" / "debug" / "debug_labels.csv"

# Читаем разметку
with open(labels_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

print(f"Всего строк в разметке: {len(rows)}")
print()

# Статистика по типам знаков
from collections import Counter
types = Counter(r["plate_type"] for r in rows)
print("Типы знаков в разметке:")
for t, n in types.most_common():
    print(f"  {t}: {n}")
print()

# Сколько уникальных изображений
images = sorted(set(r["image"] for r in rows))
print(f"Уникальных изображений с знаками: {len(images)}")
print()

# Смотрим, сколько знаков на одном изображении
per_image = Counter(r["image"] for r in rows)
multi = [(img, n) for img, n in per_image.items() if n > 1]
print(f"Изображений с несколькими знаками: {len(multi)}")
if multi:
    print("Примеры:")
    for img, n in multi[:10]:
        print(f"  {img}: {n} знаков")
print()

# Проверяем формат plate_num
print("Примеры plate_num:")
for r in rows[:10]:
    print(f"  {r['image']}: {r['plate_num']} ({r['plate_type']})")

# Проверяем, все ли изображения из разметки есть на диске
missing = [img for img in images if not (data_dir / img).exists()]
if missing:
    print()
    print(f"⚠️  Нет на диске: {len(missing)} изображений")
    for m in missing[:5]:
        print(f"  {m}")
else:
    print()
    print("Все изображения из разметки есть на диске.")
