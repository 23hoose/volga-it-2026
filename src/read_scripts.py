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
    print("=" * 70)
    print(f"{name}  ({len(lines)} строк)")
    print("=" * 70)
    # Печатаем первые 120 строк (это обычно самое важное)
    for i, line in enumerate(lines[:120], 1):
        print(f"{i:4d} | {line}")
    print()
