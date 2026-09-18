"""Рисование букв двумя руками в воздухе + распознавание нейросетью.

Как пользоваться:
- вытяни указательный палец любой руки и води им по воздуху — рисуется след
  (правая рука — зелёный, левая — бирюзовый)
- остановись с вытянутым пальцем (~1 сек) — буква с этого холста распознаётся
  и добавляется в текст, раздаётся звуковой сигнал
- кулак (все пальцы сжаты) — перестаёшь рисовать этой рукой
- руки рисуют независимо: каждая на своём холсте, можно двумя одновременно
- Esc — выход, C — очистить холсты, B — backspace

Запуск:  python3 air_write.py
"""
import time

import cv2
import mediapipe as mp
import numpy as np

import letters_net

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# суставы: большой(4), указательный(8), средний(12), безым.(16), мизинец(20)
# базы пальцев (MCP): указ -> 5, средний -> 9, безым. -> 13, мизинец -> 17
MCP = [1, 5, 9, 13, 17]
TIP = [4, 8, 12, 16, 20]
PIP = [3, 6, 10, 14, 18]

HAND_COLORS = [(0, 255, 0), (255, 200, 0)]  # правая - зелёный, левая - бирюза
HAND_NAMES = ["правой", "левой"]


def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def fingers_up(hand):
    """Палец 'вытянут', если кончик далеко от базы пальца.
    Работает при любом повороте руки (не только 'вертикально вверх')."""
    res = []
    for i in range(5):
        if i == 0:
            res.append(dist(hand[TIP[0]], hand[2]) > dist(hand[3], hand[2]) * 1.25)
        else:
            res.append(dist(hand[TIP[i]], hand[MCP[i]]) >
                       dist(hand[PIP[i]], hand[MCP[i]]) * 1.30)
    return res


def prepare_image(canvas):
    """Кадр-маска -> 28x28 нормированная картинка для нейросети."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 60)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = gray[y0:y1 + 1, x0:x1 + 1]
    side = max(crop.shape)
    sq = np.zeros((side, side), np.uint8)
    sy = (side - crop.shape[0]) // 2
    sx = (side - crop.shape[1]) // 2
    sq[sy:sy + crop.shape[0], sx:sx + crop.shape[1]] = crop
    small = cv2.resize(sq, (20, 20), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(small, 4, 4, 4, 4, cv2.BORDER_CONSTANT)
    img = np.where(img > 60, 255, 0).astype(np.float32) / 255.0
    return img.reshape(1, 784)


def beep():
    print("\a", end="")
    import sys
    sys.stdout.flush()


def new_state(h, w, color, name):
    return {
        "canvas": np.zeros((h, w, 3), np.uint8),
        "stroke": [],
        "prev_tip": None,
        "finalize_start": None,
        "color": color,
        "name": name,
        "tip": None,
        "moving": False,
        "up": None,
    }


def update_hand(state, landmarks_list, w, h):
    """Обрабатывает одну руку: след, финализация. Возвращает статусную строку."""
    color = state["color"]
    canvas = state["canvas"]
    hand = landmarks_list.landmark  # NormalizedLandmarkList -> [NormalizedLandmark]
    up = fingers_up(hand)
    state["up"] = up
    tip = (int(hand[8].x * w), int(hand[8].y * h))
    state["tip"] = tip
    moving = up[1]  # указательный вытянут = рисовать
    state["moving"] = moving

    speed = 0.0 if state["prev_tip"] is None else dist_pts(state["prev_tip"], tip)
    state["prev_tip"] = tip

    # рисуем след
    if moving:
        state["stroke"].append(tip)
        if len(state["stroke"]) > 1:
            cv2.polylines(canvas, [np.array(state["stroke"])], False, color, 8)
    else:
        state["stroke"] = []

    # финализация: вытянутый палец стоит на месте ~1 сек
    status = None
    if moving:
        if speed < 1.5:
            if state["finalize_start"] is None:
                state["finalize_start"] = time.time()
            elif time.time() - state["finalize_start"] > 1.0 and len(canvas[canvas > 0]) > 200:
                vec = prepare_image(canvas)
                pred, probs = letters_net.predict(vec)
                letter = letters_net.LETTERS[pred[0]]
                beep()
                print(f"✍ {state['name'].upper()} ({letter}, {probs[0][pred[0]]:.2f})")
                canvas[:] = 0
                state["stroke"] = []
                state["finalize_start"] = None
                status = ("recognized", letter)
            else:
                wait = int(1.0 - (time.time() - state["finalize_start"])) + 1
                status = ("finalizing", max(0, wait))
        else:
            state["finalize_start"] = None
    else:
        state["finalize_start"] = None
    return status


def process_frame(frame, both, text, hands, mp_hands, mp_draw):
    """Один кадр: detect руки, нарисовать следы, финализация букв."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    for i, state in enumerate(both):
        state["tip"] = None
        state["moving"] = False
        state["up"] = None

    detected = result.multi_hand_landmarks or []
    drawing_any = False
    fists_any = False
    finalizing_msg = None

    for i, landmarks in enumerate(detected):
        if i >= 2:
            break
        mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        status = update_hand(both[i], landmarks, w, h)
        st = both[i]
        if st["moving"]:
            drawing_any = True
        elif st["tip"] is not None and st["up"] is not None and not any(st["up"]):
            fists_any = True
        if status and status[0] == "recognized":
            text.append(status[1])
        elif status and status[0] == "finalizing":
            finalizing_msg = (st["name"], status[1])

    # накладываем следы обеих рук
    for state in both:
        mask = (state["canvas"] > 0).any(axis=2)
        frame[mask] = cv2.addWeighted(frame, 0.4, state["canvas"], 0.6, 0)[mask]
        if state["tip"]:
            cv2.circle(frame, state["tip"], 10, state["color"], -1)

    # СТАТУС-БАР: ярко показывает, рисуешь ли ты сейчас
    if finalizing_msg:
        bar, col = f"  РАСПОЗНАЮ {finalizing_msg[0].upper()}… {finalizing_msg[1]}  ", (255, 255, 0)
    elif drawing_any:
        bar, col = "  РИСУЕШЬ ✍  ", (0, 255, 0)
    elif fists_any:
        bar, col = "  КУЛАК — не рисуешь  ", (0, 0, 255)
    elif detected:
        bar, col = "  ВЫТЯНИ УКАЗАТЕЛЬНЫЙ ПАЛЕЦ  ", (0, 165, 255)
    else:
        bar, col = "  НЕТ РУК — покажи ладонь камере  ", (60, 60, 60)

    (tw, th), _ = cv2.getTextSize(bar, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.rectangle(frame, (10, 10), (10 + tw + 20, 10 + th + 20), col, -1)
    cv2.putText(frame, bar, (20, 10 + th + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # текст внизу
    line = "".join(text)
    cv2.putText(frame, line, (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 4)
    hint = ("Рисуй двумя руками: зелёный — правая, бирюзовый — левая. "
            "Остановись с пальцем на секунду — распознается.")
    cv2.putText(frame, hint, (20, h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    return None  # в случае успеха


def main():
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.6, min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Камера недоступна. Разрешение: Система > Конфиденциальность > Камера.")
        return
    for _ in range(10):
        cap.read()
    cv2.namedWindow("iaai — пишем буквы в воздухе", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("iaai — пишем буквы в воздухе", cv2.WND_PROP_TOPMOST, 1)

    text = []
    both = [None, None]
    drawing_any = False
    fists_any = False
    finalizing_msg = None
    last_error = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if both[0] is None:
            both[0] = new_state(h, w, HAND_COLORS[0], HAND_NAMES[0])
            both[1] = new_state(h, w, HAND_COLORS[1], HAND_NAMES[1])

        try:
            last_error = process_frame(frame, both, text, hands, mp_hands, mp_draw)
        except Exception as e:
            import traceback
            traceback.print_exc()
            with open("/tmp/iaai_error.log", "a") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                traceback.print_exc(file=f)
            last_error = repr(e)

        if last_error:
            cv2.putText(frame, "Ошибка: " + last_error, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("iaai — пишем буквы в воздухе", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("c"):
            for st in both:
                st["canvas"][:] = 0
                st["stroke"] = []
        if key == ord("b") and text:
            text.pop()

    cap.release()
    cv2.destroyAllWindows()
    print("Набрано:", "".join(text))


def dist_pts(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


if __name__ == "__main__":
    main()