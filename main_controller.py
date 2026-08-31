import cv2
import pydirectinput
from HandTrackingModule import HandDetector
import time

# Zero out PyDirectInput internal delays
pydirectinput.PAUSE = 0.0
pydirectinput.FAILSAFE = False

# Fast 480p capture pipeline for maximum FPS
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

detector = HandDetector(max_hands=1)

active_keys = set()


def press(key):
    if key not in active_keys:
        pydirectinput.keyDown(key)
        active_keys.add(key)


def release(key):
    if key in active_keys:
        pydirectinput.keyUp(key)
        active_keys.remove(key)


def release_all():
    for k in list(active_keys):
        pydirectinput.keyUp(k)
    active_keys.clear()


p_time = 0

print("\n=== INSTANT VIRTUAL D-PAD CONTROLLER ===")
print("Move hand into the on-screen zones to instantly trigger keys.")
print("Pinch (Thumb + Index) for SPACE / NITRO.")
print("Press 'q' to quit.\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    curr_time = time.time()

    # Process frame
    detector.find_hands(frame)
    data = detector.get_hand_center_and_pinch(frame)

    # UI Box Boundaries (Center Neutral Zone)
    cx_min, cx_max = int(w * 0.35), int(w * 0.65)
    cy_min, cy_max = int(h * 0.35), int(h * 0.65)

    # Draw Instant Interactive Grid
    cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 255), 2)
    cv2.putText(
        frame,
        "NEUTRAL ZONE",
        (cx_min + 10, cy_min + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    # Zone Labels
    cv2.putText(
        frame,
        "UP / GAS",
        (w // 2 - 35, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "DOWN / BRAKE",
        (w // 2 - 55, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        frame, "LEFT", (15, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
    )
    cv2.putText(
        frame,
        "RIGHT",
        (w - 70, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    if data is None:
        release_all()
    else:
        hx, hy, is_pinching, (tx, ty, ix, iy) = data

        # Draw glowing tracking cursor
        cv2.circle(frame, (hx, hy), 12, (0, 255, 255), cv2.FILLED)

        # 1. INSTANT PINCH (SPACE)
        if is_pinching:
            press("space")
            cv2.line(frame, (tx, ty), (ix, iy), (0, 0, 255), 4)
            cv2.putText(
                frame,
                "NITRO / JUMP!",
                (hx - 40, hy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        else:
            release("space")

        # 2. INSTANT HORIZONTAL STEERING
        if hx < cx_min:
            press("left")
            release("right")
            cv2.rectangle(frame, (0, 0), (cx_min, h), (0, 255, 255), 3)
        elif hx > cx_max:
            press("right")
            release("left")
            cv2.rectangle(frame, (cx_max, 0), (w, h), (0, 255, 255), 3)
        else:
            release("left")
            release("right")

        # 3. INSTANT VERTICAL THROTTLE / BRAKE
        if hy < cy_min:
            press("up")
            release("down")
            cv2.rectangle(frame, (0, 0), (w, cy_min), (0, 255, 0), 3)
        elif hy > cy_max:
            press("down")
            release("up")
            cv2.rectangle(frame, (0, cy_max), (w, h), (0, 0, 255), 3)
        else:
            release("up")
            release("down")

    # FPS Counter
    fps = 1 / (curr_time - p_time) if (curr_time - p_time) > 0 else 0
    p_time = curr_time
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 25),
        cv2.FONT_HERSHEY_PLAIN,
        1.4,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Instant Virtual D-Pad", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

release_all()
cap.release()
cv2.destroyAllWindows()
