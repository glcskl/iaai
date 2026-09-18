"""Рисование букв пальцем в воздухе + распознавание нейросетью.

Как пользоваться:
- вытяни указательный палец и води им по воздуху — рисуется зелёный след
- остановись с вытянутым пальцем (~1 сек) — буква распознаётся и
  добавляется в текст, раздаётся звуковой сигнал
- кулак (все пальцы сжаты) — перестаёшь рисовать
- Esc — выход, C — очистить холст, B — backspace

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


def main():
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
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

    stroke = []                 # точки текущего штриха
    canvas = None               # маска следа поверх кадра
    finalize_start = None       # когда палец вытянут и неподвижен
    prev_tip = None
    text = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if canvas is None:
            canvas = np.zeros((h, w, 3), np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        tip = None
        moving = False
        status = "Нет руки — покажи ладонь камере"
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            up = fingers_up(hand.landmark)
            tip = (int(hand.landmark[8].x * w), int(hand.landmark[8].y * h))
            moving = up[1]  # указательный вытянут = рисовать
            if moving:
                status = "✍ РИСУЕШЬ — води пальцем по воздуху"
            elif not any(up):
                status = "✋ КУЛАК — рисование остановлено"
            else:
                status = "Палец согнут — вытяни указательный"

        # движение пальца (для финализации по неподвижности)
        if tip:
            moving_speed = 0.0 if prev_tip is None else dist_pts(prev_tip, tip)
            prev_tip = tip
        else:
            moving_speed = 0.0
            prev_tip = None

        # рисуем след
        if tip and moving:
            stroke.append(tip)
            if len(stroke) > 1:
                cv2.polylines(canvas, [np.array(stroke)], False, (0, 255, 0), 8)
        else:
            stroke = []

        # финализация: вытянутый палец стоит на месте ~1 сек
        if tip and moving:
            if moving_speed < 1.5:
                if finalize_start is None:
                    finalize_start = time.time()
                elif time.time() - finalize_start > 1.0 and len(canvas[canvas > 0]) > 200:
                    vec = prepare_image(canvas)
                    pred, probs = letters_net.predict(vec)
                    letter = letters_net.LETTERS[pred[0]]
                    text.append(letter)
                    beep()
                    print("✍", letter, f"({probs[0][pred[0]]:.2f})")
                    canvas[:] = 0
                    stroke = []
                    finalize_start = None
                else:
                    # показываем таймер финализации
                    status = f"Распознаю… {max(0, int(1.0 - (time.time() - finalize_start)))+1}"
            else:
                finalize_start = None
        else:
            finalize_start = None

        # накладываем след
        mask = (canvas > 0).any(axis=2)
        frame[mask] = cv2.addWeighted(frame, 0.4, canvas, 0.6, 0)[mask]

        # курсор-кончик пальца
        if tip:
            cv2.circle(frame, tip, 10, (0, 255, 255), -1)

        # СТАТУС-БАР: вид ярко говорит, рисуешь ли ты сейчас
        if tip and moving:
            bar, col = "  РИСУЕШЬ  ", (0, 255, 0)
        elif tip and not any(up):
            bar, col = "  КУЛАК — не рисуешь  ", (0, 0, 255)
        else:
            bar, col = "  ПАЛЕЦ НЕ ВЫТЯНУТ  ", (0, 165, 255)

        (tw, th), _ = cv2.getTextSize(bar, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(frame, (10, 10), (10 + tw + 20, 10 + th + 20), col, -1)
        cv2.putText(frame, bar, (20, 10 + th + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        # текст внизу
        line = "".join(text)
        cv2.putText(frame, line, (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 4)
        hint = "Вытяни указательный палец и рисуй в воздухе. Остановись на секунду — распознается."
        cv2.putText(frame, hint, (20, h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        cv2.imshow("iaai — пишем буквы в воздухе", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("c"):
            canvas[:] = 0
            stroke = []
        if key == ord("b") and text:
            text.pop()

    cap.release()
    cv2.destroyAllWindows()
    print("Набрано:", "".join(text))


def dist_pts(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


if __name__ == "__main__":
    main()