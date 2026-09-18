"""Считалка чисел руками через веб-камеру MacBook.

Показываешь ладонь в камеру — скрипт считает вытянутые пальцы:
0..5 одной рукой, до 10 двумя руками. История чисел пишется внизу экрана.

Запуск:
    python3 finger_counter.py

Выход: Esc.  Сброс истории: R.
Требует разрешения на камеру (система сама спросит при первом запуске).
"""
import time

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Индексы кончиков пальцев и суставов из 21 точки MediaPipe
TIP = [4, 8, 12, 16, 20]   # кончики: большой, указательный, средний, безымянный, мизинец
PIP = [3, 6, 10, 14, 18]   # средние суставы (для большого — отдельный критерий)


def is_finger_up(hand_landmarks, finger_idx):
    """Пальцы 1-4: вытянут, если кончик выше сустава (меньше y).
    Большой палец: направлен вбок — для правой руки кончик правее сустава."""
    if finger_idx == 0:  # большой
        return hand_landmarks[TIP[0]].x > hand_landmarks[PIP[0]].x
    return hand_landmarks[TIP[finger_idx]].y < hand_landmarks[PIP[finger_idx]].y


def count_fingers(hand_landmarks):
    return sum(is_finger_up(hand_landmarks, i) for i in range(5))


def draw_dotted_rect(img, x, y, w, h, color, dot=6, gap=6):
    """Точечная рамка вокруг числа."""
    pts = []
    for i in range(x, x + w, dot + gap):
        pts.append((i, y))
    for i in range(x, x + w, dot + gap):
        pts.append((i, y + h))
    for i in range(y, y + h, dot + gap):
        pts.append((x, i))
    for i in range(y, y + h, dot + gap):
        pts.append((x + w, i))
    for px, py in pts:
        cv2.circle(img, (px, py), 3, color, -1)


def main():
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру. "
              "Проверь разрешения: Системные настройки > Конфиденциальность > Камера.")
        return
    # даём камере пару секунд на «прогрев» и запрос разрешения macOS
    for _ in range(10):
        cap.read()
    cv2.namedWindow("Считаем руками", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Считаем руками", cv2.WND_PROP_TOPMOST, 1)

    history = []
    last_change = time.time()
    last_total = -1

    print("Показывай числа руками! Esc — выход, R — очистить историю.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # зеркало, как привычно
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        total = 0
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                n = count_fingers(hand.landmark)
                total += n
                x, y = int(hand.landmark[0].x * frame.shape[1]), int(hand.landmark[0].y * frame.shape[0])
                cv2.putText(frame, str(n), (x - 10, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # число попадает в историю, если стабильно держится 0.3 сек
        if total != last_total:
            last_total = total
            last_change = time.time()
        elif (not history or history[-1] != total) and time.time() - last_change > 0.3:
            history.append(total)

        # КРАСИВЫЙ ОВЕРЛЕЙ: полупрозрачная «плёнка» внизу для читаемости
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # БОЛЬШОЕ ЧИСЛО по центру + ЗЕЛЁНЫЕ ТОЧКИ вокруг него
        label = f"{total}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 6)
        tx = (w - tw) // 2
        ty = h - 40
        cv2.putText(frame, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 6)
        pad = 16
        draw_dotted_rect(frame, tx - pad, ty - th - pad,
                         tw + 2 * pad, th + 2 * pad, (0, 255, 0))

        big = " ".join(map(str, history[-12:]))
        cv2.putText(frame, "История: " + big, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Считаем руками", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            break
        if key == ord("r"):
            history = []

    cap.release()
    cv2.destroyAllWindows()
    print("История чисел:", history)


if __name__ == "__main__":
    main()