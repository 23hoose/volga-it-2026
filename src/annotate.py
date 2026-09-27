"""Простой скрипт разметки bbox в YOLO-формате.

Управление:
  ЛКМ: нарисовать прямоугольник вокруг номера
  ЦИФРА 1/2/3/4: назначить класс (1=type1, 2=type1a, 3=type1b, 4=other)
  U: отменить последний прямоугольник на текущем изображении
  S: сохранить и перейти к следующему
  Q или ESC: выход (сохраняет текущий)
  P: пропустить изображение

Разметка сохраняется в data/debug/labels_yolo/<image_stem>.txt
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "debug" / "debug set"
OUT_DIR = ROOT / "data" / "debug" / "labels_yolo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {1: "type1", 2: "type1a", 3: "type1b", 4: "other"}
CLASS_IDS = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}

# Состояние на текущее изображение
state = {
    "drawing": False,
    "start": None,
    "current_rect": None,
    "rects": [],   # список (x1, y1, x2, y2, class_id)
    "img": None,
    "orig": None,
    "path": None,
}


def draw_all():
    """Перерисовывает окно с учётом текущего состояния."""
    disp = state["orig"].copy()

    # Уже нарисованные прямоугольники
    for (x1, y1, x2, y2, cid) in state["rects"]:
        color = [(0, 200, 0), (0, 165, 255), (255, 0, 0), (128, 128, 128)][cid]
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        label = list(CLASS_IDS.keys())[list(CLASS_IDS.values()).index(cid)]
        cv2.putText(disp, label, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Текущий прямоугольник (в процессе рисования)
    if state["drawing"] and state["current_rect"] is not None:
        x1, y1, x2, y2 = state["current_rect"]
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Подсказки
    hint = f"{state['path'].name} | Прямоугольников: {len(state['rects'])}"
    cv2.putText(disp, hint, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(disp, "1=type1 2=type1a 3=type1b 4=other | S=save+next U=undo P=skip Q=quit",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    state["img"] = disp
    cv2.imshow("Annotation", disp)


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing"] = True
        state["start"] = (x, y)
        state["current_rect"] = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
        x1, y1 = state["start"]
        state["current_rect"] = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
        draw_all()
    elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
        state["drawing"] = False
        x1, y1 = state["start"]
        x1, y1, x2, y2 = min(x1, x), min(y1, y), max(x1, x), max(y1, y)
        state["current_rect"] = None
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            # слишком маленький — игнор
            draw_all()
            return
        # Спрашиваем класс
        print(f"\nНарисован прямоугольник: ({x1},{y1})-({x2},{y2})")
        print("Введи класс в консоли: 1=type1, 2=type1a, 3=type1b, 4=other")
        # По умолчанию — 2 (type1a), самый частый
        ans = input("Класс [по умолч. 2]: ").strip()
        if ans == "":
            cid = 1
        else:
            try:
                n = int(ans)
                if n not in CLASS_NAMES:
                    n = 2
                cid = n - 1
            except ValueError:
                cid = 1
        state["rects"].append((x1, y1, x2, y2, cid))
        draw_all()


def save_labels(path, rects, img_w, img_h):
    """Сохраняет разметку в YOLO-формате."""
    out_txt = OUT_DIR / (path.stem + ".txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for (x1, y1, x2, y2, cid) in rects:
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    print(f"Сохранено: {out_txt}")


def main():
    files = sorted(IMG_DIR.glob("*.jpg"))
    print(f"Изображений: {len(files)}")
    print("Разметка сохраняется в:", OUT_DIR)
    print()

    cv2.namedWindow("Annotation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotation", mouse_cb)

    idx = 0
    while idx < len(files):
        path = files[idx]
        img = cv2.imread(str(path))
        if img is None:
            print(f"Не читается: {path.name}")
            idx += 1
            continue

        state["orig"] = img.copy()
        state["img"] = img
        state["path"] = path
        state["rects"] = []
        state["drawing"] = False
        state["start"] = None
        state["current_rect"] = None

        print(f"\n[{idx + 1}/{len(files)}] {path.name}")
        print("  Размечай. S — сохранить и перейти дальше. P — пропустить. Q — выход.")

        draw_all()

        action = None
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:  # Q или ESC
                action = "quit"
                break
            elif key == ord('s'):
                action = "save"
                break
            elif key == ord('p'):
                action = "skip"
                break
            elif key == ord('u'):
                if state["rects"]:
                    state["rects"].pop()
                    draw_all()

        if action == "quit":
            if state["rects"]:
                save_labels(path, state["rects"], img.shape[1], img.shape[0])
            break
        elif action == "save":
            save_labels(path, state["rects"], img.shape[1], img.shape[0])
            idx += 1
        elif action == "skip":
            print("  Пропущено")
            idx += 1

    cv2.destroyAllWindows()
    print("\nГотово.")


if __name__ == "__main__":
    main()