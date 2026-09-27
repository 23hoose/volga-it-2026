import csv
from collections import Counter
from pathlib import Path

meta = Path("dataset/meta.csv")
with open(meta, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

types = Counter(r["plate_type"] for r in rows)
print("Всего строк:", len(rows))
for t, n in types.most_common():
    print(f"  {t}: {n}")
