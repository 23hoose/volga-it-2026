from pathlib import Path

root = Path(__file__).resolve().parent.parent
debug = root / "data" / "debug"

for name in ["evaluate.py", "validate_dataset.py"]:
    p = debug / name
    if not p.exists():
        print(f"Нет файла: {p}")
        continue
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    out = root / f"{name}.utf8.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"{name}  ({len(lines)} строк)\n")
        f.write("=" * 70 + "\n")
        for i, line in enumerate(lines, 1):
            f.write(f"{i:4d} | {line}\n")
    print(f"Сохранено: {out}")
