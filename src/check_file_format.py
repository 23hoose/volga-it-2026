"""Проверка: что за формат у файлов (по магическим байтам)."""
from pathlib import Path

BG_DIR = Path(__file__).resolve().parent.parent / "dataset" / "generator" / "backgrounds"

files = sorted(BG_DIR.glob("*"))
for f in files:
    with open(f, "rb") as fh:
        head = fh.read(16)
    hex_head = head.hex()
    ascii_head = "".join(chr(b) if 32 <= b < 127 else "." for b in head)
    print(f"{f.name}:")
    print(f"  hex:   {hex_head}")
    print(f"  ascii: {ascii_head}")