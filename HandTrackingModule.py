import os
import urllib.request
import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 21 Hand landmark connection pairs for drawing
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    (0, 17),  # Palm Base
]

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model_downloaded():
    """Downloads the official MediaPipe hand landmarker model if missing."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe hand tracking model (~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")


class HandDetector:
    def __init__(
        self, max_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5
    ):
        ensure_model_downloaded()
        self.max_hands = max_hands
        self.tip_ids = [4, 8, 12, 16, 20]

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img, draw=True):
        """Processes video frame and renders hand landmark skeletons."""
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        frame_timestamp_ms = int(time.time() * 1000)

        self.results = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if self.results and self.results.hand_landmarks and draw:
            for hand_landmarks in self.results.hand_landmarks:
                # Draw lines between joints
                for start_idx, end_idx in HAND_CONNECTIONS:
                    pt1 = (
                        int(hand_landmarks[start_idx].x * w),
                        int(hand_landmarks[start_idx].y * h),
                    )
                    pt2 = (
                        int(hand_landmarks[end_idx].x * w),
                        int(hand_landmarks[end_idx].y * h),
                    )
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)

                # Draw joint points
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), cv2.FILLED)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        """Returns list of [id, x, y] for all 21 keypoints of the requested hand."""
        landmark_list = []
        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                hand_landmarks = self.results.hand_landmarks[hand_no]
                h, w, _ = img.shape
                for id_val, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id_val, cx, cy])
                    if draw and id_val in self.tip_ids:
                        cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
        return landmark_list


def main():
    p_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_hands=2)

    print("Camera running. Show your hand to the camera! Press 'q' to exit.")

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = detector.find_hands(img, draw=True)
        lm_list = detector.find_position(img, hand_no=0, draw=True)

        if len(lm_list) != 0:
            print(f"Index Finger Tip: X={lm_list[8][1]}, Y={lm_list[8][2]}")

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 40),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Hand Tracking - Modern MediaPipe", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
