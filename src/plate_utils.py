"""Утилиты для работы с российскими ГРЗ.

Используется и при генерации датасета, и в inference, и в постобработке.
Только стандартная библиотека Python.
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Справочники формата (ГОСТ Р 50577-2018)
# ---------------------------------------------------------------------------

# Буквы, совпадающие по начертанию в кириллице и латинице.
LATIN_LETTERS = "ABEKMHOPCTYX"
CYR_LETTERS = "АВЕКМНОРСТУХ"

CYR_TO_LAT = str.maketrans(
    CYR_LETTERS + CYR_LETTERS.lower(),
    LATIN_LETTERS + LATIN_LETTERS,
)

# Типы знаков.
PLATE_TYPES = ("type1", "type1a", "type1b", "other")
TARGET_TYPES = ("type1", "type1a", "type1b")

# Маски. "#" — нераспознанная позиция.
_L = f"[{LATIN_LETTERS}#]"
_D = "[0-9#]"
_REGION = f"{_D}{{2,3}}"

PLATE_MASKS = {
    # Л ЦЦЦ ЛЛ + регион (2 или 3 цифры) — типы 1 и 1А
    "type1":  re.compile(f"^{_L}{_D}{{3}}{_L}{{2}}{_REGION}$"),
    "type1a": re.compile(f"^{_L}{_D}{{3}}{_L}{{2}}{_REGION}$"),
    # ЛЛ ЦЦЦ + регион — тип 1Б (такси)
    "type1b": re.compile(f"^{_L}{{2}}{_D}{{3}}{_REGION}$"),
    # other — произвольная строка из A-Z, 0-9, #, до 12 символов
    "other":  re.compile(r"^[A-Z0-9#]{0,12}$"),
}

# Длина "тела" без региона.
BODY_LEN = {
    "type1": 6,
    "type1a": 6,
    "type1b": 5,
}

# Классы для YOLO-labels.
LABEL_CLASSES = {0: "type1", 1: "type1a", 2: "type1b", 3: "other"}
CLASS_TO_ID = {v: k for k, v in LABEL_CLASSES.items()}


# ---------------------------------------------------------------------------
# Нормализация и проверка
# ---------------------------------------------------------------------------

def normalize_plate(text: str) -> str:
    """Приводит номер к канонической форме:
    - заменяет кириллицу на латиницу,
    - приводит к верхнему регистру,
    - убирает пробелы и разделители.
    """
    if not text:
        return ""
    text = text.translate(CYR_TO_LAT)
    text = text.upper()
    # Убираем всё, кроме A-Z, 0-9, #.
    text = re.sub(r"[^A-Z0-9#]", "", text)
    return text


def check_plate(plate: str, plate_type: str) -> bool:
    """Проверяет, соответствует ли номер маске типа.
    Возвращает True, если ок.
    """
    if plate_type not in PLATE_MASKS:
        return False
    return bool(PLATE_MASKS[plate_type].fullmatch(plate))


def plate_problems(plate: str, plate_type: str) -> list[str]:
    """Возвращает список проблем с номером (пустой — всё ок).
    Полезно для отчётов и отладки.
    """
    problems = []
    if plate != plate.strip():
        problems.append("пробелы по краям")
    if any(ch in CYR_LETTERS + CYR_LETTERS.lower() for ch in plate):
        problems.append("кириллица (нужна латиница)")
    if plate != plate.upper():
        problems.append("не в верхнем регистре")
    if not check_plate(plate, plate_type):
        problems.append(f"не соответствует маске {plate_type}")
    return problems


def readable_count(plate: str) -> int:
    """Сколько в номере читаемых символов (не #)."""
    return sum(1 for ch in plate if ch != "#")


def is_fully_readable(plate: str) -> bool:
    return "#" not in plate


def split_body_region(plate: str, plate_type: str) -> tuple[str, str]:
    """Разделяет номер на тело и регион по длине body.
    Для type1b body=5, для type1/type1a body=6.
    """
    body_len = BODY_LEN.get(plate_type)
    if body_len is None or len(plate) <= body_len:
        return plate, ""
    return plate[:body_len], plate[body_len:]


# ---------------------------------------------------------------------------
# Генерация (для синтетики)
# ---------------------------------------------------------------------------

def make_random_plate(plate_type: str, rng) -> str:
    """Генерирует случайный валидный номер заданного типа.
    rng — объект random.Random (для воспроизводимости).
    """
    if plate_type not in TARGET_TYPES:
        raise ValueError(f"Нельзя сгенерировать номер типа {plate_type}")

    def L():
        return rng.choice(LATIN_LETTERS)

    def D():
        return rng.choice("0123456789")

    region_len = rng.choice([2, 3])
    region = "".join(D() for _ in range(region_len))

    if plate_type == "type1b":
        # ЛЛ ЦЦЦ + регион
        return f"{L()}{L()}{D()}{D()}{D()}{region}"
    else:
        # Л ЦЦЦ ЛЛ + регион
        return f"{L()}{D()}{D()}{D()}{L()}{L()}{region}"


# ---------------------------------------------------------------------------
# Быстрая проверка при запуске модуля
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Тесты
    assert normalize_plate("а123вс777") == "A123BC777"
    assert normalize_plate("A 123 BC 777") == "A123BC777"
    assert normalize_plate("АВ12377") == "AB12377"

    assert check_plate("A123BC777", "type1")
    assert check_plate("A123BC77", "type1a")
    assert check_plate("AB12377", "type1b")
    assert check_plate("AB123777", "type1b")
    assert not check_plate("A123BC777", "type1b")
    assert not check_plate("AB12377", "type1")

    assert readable_count("A##3BC###") == 4
    assert not is_fully_readable("A##3BC###")

    body, region = split_body_region("A123BC777", "type1")
    assert body == "A123BC" and region == "777"

    body, region = split_body_region("AB12377", "type1b")
    assert body == "AB123" and region == "77"

    import random
    rng = random.Random(42)
    for _ in range(5):
        p1 = make_random_plate("type1a", rng)
        assert check_plate(p1, "type1a"), p1
        p2 = make_random_plate("type1b", rng)
        assert check_plate(p2, "type1b"), p2

    print("Все тесты plate_utils.py пройдены.")
