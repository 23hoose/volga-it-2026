from pathlib import Path

p = Path(__file__).resolve().parent.parent / "data" / "debug" / "README.md"
text = p.read_text(encoding="utf-8")
print(text)
