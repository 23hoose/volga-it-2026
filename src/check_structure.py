from pathlib import Path

# Ищем корень проекта — папку, где лежит venv
here = Path(__file__).resolve().parent
root = here.parent  # из src/ поднимаемся на уровень выше

print('=== Проверка структуры проекта ===')
print(f'Корень: {root}')
print()

expected = [
    'src', 'data', 'models', 'dataset',
    'dataset/images', 'dataset/images/real', 'dataset/images/synthetic',
    'dataset/labels', 'dataset/generator', 'venv',
]

all_ok = True
for p in expected:
    full = root / p
    exists = full.exists()
    mark = '[OK]  ' if exists else '[НЕТ] '
    print(mark, p)
    if not exists:
        all_ok = False

print()
print('Все папки на месте!' if all_ok else 'Некоторые папки отсутствуют.')
