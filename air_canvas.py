import cv2
import numpy as np
import mediapipe as mp
import math
import os
import warnings
warnings.filterwarnings("ignore")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    static_image_mode=False,
    model_complexity=0
)

prev_x, prev_y = 0, 0
color = (255, 0, 0)
eraser_mode = False
save_count = 1

# ✅ Function to check which fingers are up
def finger_states(hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand.landmark[tips[0]].x < hand.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in tips[1:]:
        if hand.landmark[tip].y < hand.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# ✅ Distance helper
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    # ✅ UI Buttons
    cv2.rectangle(frame, (10, 10), (60, 60), (255, 0, 0), -1)
    cv2.rectangle(frame, (70, 10), (120, 60), (0, 255, 0), -1)
    cv2.rectangle(frame, (130, 10), (180, 60), (0, 0, 255), -1)

    cv2.rectangle(frame, (200, 10), (280, 60), (50, 50, 50), -1)
    cv2.putText(frame, "Clear", (205, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.rectangle(frame, (300, 10), (380, 60), (0, 255, 255), -1)
    cv2.putText(frame, "Save", (305, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        h, w, _ = frame.shape

        x1 = int(hand.landmark[8].x * w)
        y1 = int(hand.landmark[8].y * h)

        x2 = int(hand.landmark[4].x * w)
        y2 = int(hand.landmark[4].y * h)

        fingers = finger_states(hand)

        # ✅ Gestures
        draw_mode = (fingers == [0,1,0,0,0])      # Index finger only
        stop_mode = (fingers == [0,0,0,0,0])      # Fist

        # ✅ Eraser (pinch)
        if distance((x1, y1), (x2, y2)) < 30:
            eraser_mode = True
        else:
            eraser_mode = False

        # ✅ Button Actions
        if 10 < x1 < 60 and 10 < y1 < 60:
            color = (255, 0, 0)
        if 70 < x1 < 120 and 10 < y1 < 60:
            color = (0, 255, 0)
        if 130 < x1 < 180 and 10 < y1 < 60:
            color = (0, 0, 255)
        if 200 < x1 < 280 and 10 < y1 < 60:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # ✅ Save
        if 300 < x1 < 380 and 10 < y1 < 60:
            filename = f"drawing_{save_count}.png"
            cv2.imwrite(filename, canvas)
            print(f"✅ Saved: {filename}")
            save_count += 1

        # ✅ STOP mode (fist)
        if stop_mode:
            prev_x, prev_y = 0, 0
            cv2.putText(frame, "STOP", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # ✅ DRAW mode (index finger only)
        elif draw_mode:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x1, y1

            if eraser_mode:
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), (0,0,0), 40)
            else:
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), color, 5)

            prev_x, prev_y = x1, y1

        else:
            prev_x, prev_y = 0, 0

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    else:
        prev_x, prev_y = 0, 0

    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Air Canvas Pro", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
