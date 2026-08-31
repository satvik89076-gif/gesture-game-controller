import cv2
import numpy as np
import time
import math
import pydirectinput
from HandTrackingModule import HandDetector

# Minimize PyDirectInput latency
pydirectinput.PAUSE = 0.001

# Initialize camera & detector
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = HandDetector(
    max_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5
)

# Key state tracking
current_keys_down = set()


def press_key(key):
    """Presses and holds a key if not already held."""
    if key not in current_keys_down:
        pydirectinput.keyDown(key)
        current_keys_down.add(key)


def release_key(key):
    """Releases a key if currently pressed."""
    if key in current_keys_down:
        pydirectinput.keyUp(key)
        current_keys_down.remove(key)


def tap_key(key):
    """Triggers an instantaneous key press for swipe actions."""
    pydirectinput.press(key)


def release_all_keys():
    """Releases all active held keys safely."""
    for key in list(current_keys_down):
        pydirectinput.keyUp(key)
    current_keys_down.clear()


def is_finger_up(landmarks, tip_id, pip_id):
    """Checks if a finger is extended based on vertical landmarks."""
    return landmarks[tip_id][2] < landmarks[pip_id][2]


def calculate_distance(p1, p2):
    """Calculates Euclidean pixel distance between two landmark points."""
    return math.hypot(p2[1] - p1[1], p2[2] - p1[2])


# Swipe detection tracker variables
prev_wrist_pos = None
prev_wrist_time = 0
last_swipe_time = 0
SWIPE_COOLDOWN = 0.35  # Cooldown in seconds to prevent accidental double-swipes
SWIPE_VELOCITY_THRESH = 750  # Minimum pixel speed per second to count as a swipe

last_detected_swipe = "NONE"
swipe_display_timer = 0

p_time = 0

print("\n=== Virtual Controller with Swipe + Continuous Mechanics ===")
print("Swipe TOP: Jump (SPACE)")
print("Swipe LEFT: Turn Left")
print("Swipe RIGHT: Turn Right")
print("Pinch (Thumb + Index): Nitro (SPACE)")
print("Hold Palm Open: Continuous Gas (UP)")
print("Closed Fist: Continuous Brake (DOWN)")
print("Press 'q' in video window to exit.\n")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror feed
    h, w, _ = img.shape
    curr_time = time.time()

    img = detector.find_hands(img, draw=True)
    hand_count = (
        len(detector.results.hand_landmarks)
        if (detector.results and detector.results.hand_landmarks)
        else 0
    )

    action_text = "IDLE"
    steer_text = "CENTER"
    nitro_text = "OFF"

    if hand_count == 0:
        release_all_keys()
        prev_wrist_pos = None

    elif hand_count == 1:
        # SINGLE HAND MODE
        lm = detector.find_position(img, hand_no=0, draw=False)
        if len(lm) >= 21:
            wrist_x, wrist_y = lm[0][1], lm[0][2]

            # --- SWIPE DETECTION LOGIC ---
            if prev_wrist_pos is not None:
                dt = curr_time - prev_wrist_time
                if dt > 0:
                    vx = (wrist_x - prev_wrist_pos[0]) / dt
                    vy = (wrist_y - prev_wrist_pos[1]) / dt

                    if curr_time - last_swipe_time > SWIPE_COOLDOWN:
                        # Swipe Up / Top (Jump) -> vy is negative (screen coords go down)
                        if vy < -SWIPE_VELOCITY_THRESH and abs(vy) > abs(vx) * 1.2:
                            tap_key("space")
                            last_detected_swipe = "SWIPE TOP (JUMP)"
                            last_swipe_time = curr_time
                            swipe_display_timer = curr_time + 0.6
                        # Swipe Right
                        elif vx > SWIPE_VELOCITY_THRESH and abs(vx) > abs(vy) * 1.2:
                            tap_key("right")
                            last_detected_swipe = "SWIPE RIGHT"
                            last_swipe_time = curr_time
                            swipe_display_timer = curr_time + 0.6
                        # Swipe Left
                        elif vx < -SWIPE_VELOCITY_THRESH and abs(vx) > abs(vy) * 1.2:
                            tap_key("left")
                            last_detected_swipe = "SWIPE LEFT"
                            last_swipe_time = curr_time
                            swipe_display_timer = curr_time + 0.6

            prev_wrist_pos = (wrist_x, wrist_y)
            prev_wrist_time = curr_time

            # --- PINCH NITRO DETECTION ---
            pinch_dist = calculate_distance(lm[4], lm[8])
            if pinch_dist < 35:
                press_key("space")
                nitro_text = "ACTIVE (SPACE)"
                cx = (lm[4][1] + lm[8][1]) // 2
                cy = (lm[4][2] + lm[8][2]) // 2
                cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
            else:
                release_key("space")
                nitro_text = "OFF"

            # --- CONTINUOUS ACCEL / BRAKE ---
            index_up = is_finger_up(lm, 8, 6)
            middle_up = is_finger_up(lm, 12, 10)
            ring_up = is_finger_up(lm, 16, 14)
            pinky_up = is_finger_up(lm, 20, 18)
            fingers_up_count = sum([index_up, middle_up, ring_up, pinky_up])

            if fingers_up_count >= 3:
                press_key("up")
                release_key("down")
                action_text = "GAS / ACCEL (UP)"
            elif fingers_up_count == 0:
                press_key("down")
                release_key("up")
                action_text = "BRAKE (DOWN)"
            else:
                release_key("up")
                release_key("down")
                action_text = "COASTING"

            # --- CONTINUOUS STEERING POSITION ---
            center_x = w // 2
            deadzone = 60
            if wrist_x < center_x - deadzone:
                press_key("left")
                release_key("right")
                steer_text = "STEER LEFT"
            elif wrist_x > center_x + deadzone:
                press_key("right")
                release_key("left")
                steer_text = "STEER RIGHT"
            else:
                release_key("left")
                release_key("right")
                steer_text = "CENTER"

    elif hand_count >= 2:
        # DUAL HAND MODE
        hand1_lm = detector.find_position(img, hand_no=0, draw=False)
        hand2_lm = detector.find_position(img, hand_no=1, draw=False)

        if len(hand1_lm) >= 21 and len(hand2_lm) >= 21:
            w1_x, w1_y = hand1_lm[0][1], hand1_lm[0][2]
            w2_x, w2_y = hand2_lm[0][1], hand2_lm[0][2]

            pinch1 = calculate_distance(hand1_lm[4], hand1_lm[8])
            pinch2 = calculate_distance(hand2_lm[4], hand2_lm[8])

            if pinch1 < 35 or pinch2 < 35:
                press_key("space")
                nitro_text = "ACTIVE (SPACE)"
            else:
                release_key("space")
                nitro_text = "OFF"

            if w1_x < w2_x:
                left_y, right_y = w1_y, w2_y
            else:
                left_y, right_y = w2_y, w1_y

            cv2.line(
                img, (int(w1_x), int(w1_y)), (int(w2_x), int(w2_y)), (255, 255, 0), 3
            )
            tilt_diff = left_y - right_y

            if tilt_diff > 45:
                press_key("right")
                release_key("left")
                steer_text = "STEER RIGHT"
            elif tilt_diff < -45:
                press_key("left")
                release_key("right")
                steer_text = "STEER LEFT"
            else:
                release_key("left")
                release_key("right")
                steer_text = "CENTER"

            h1_open = is_finger_up(hand1_lm, 8, 6) and is_finger_up(hand1_lm, 12, 10)
            h2_open = is_finger_up(hand2_lm, 8, 6) and is_finger_up(hand2_lm, 12, 10)

            if h1_open or h2_open:
                press_key("up")
                release_key("down")
                action_text = "GAS / ACCEL (UP)"
            else:
                press_key("down")
                release_key("up")
                action_text = "BRAKE (DOWN)"

    # Heads-Up Display (HUD)
    cv2.rectangle(img, (10, 10), (480, 155), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        img,
        f"ACTION: {action_text}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"STEER : {steer_text}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        img,
        f"NITRO : {nitro_text}",
        (20, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 165, 255),
        2,
    )

    swipe_text = last_detected_swipe if curr_time < swipe_display_timer else "NONE"
    swipe_color = (0, 255, 255) if swipe_text != "NONE" else (160, 160, 160)
    cv2.putText(
        img,
        f"SWIPE : {swipe_text}",
        (20, 134),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        swipe_color,
        2,
    )

    fps = 1 / (curr_time - p_time) if (curr_time - p_time) > 0 else 0
    p_time = curr_time
    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (w - 120, 35),
        cv2.FONT_HERSHEY_PLAIN,
        1.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Virtual Game Controller HUD", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

release_all_keys()
cap.release()
cv2.destroyAllWindows()
