import os
import urllib.request
import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class HandDetector:
    def __init__(
        self, max_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ):
        ensure_model_downloaded()
        self.max_hands = max_hands

        # Fast single-hand real-time tracking
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img):
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int(time.time() * 1000)

        self.results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return img

    def get_hand_center_and_pinch(self, img):
        """Returns (x, y, is_pinching) in constant time for instant response."""
        if self.results and self.results.hand_landmarks:
            lm = self.results.hand_landmarks[0]
            h, w, _ = img.shape

            # Landmark 9 is the palm center (knuckle of middle finger) - very stable
            center_x = int(lm[9].x * w)
            center_y = int(lm[9].y * h)

            # Thumb Tip (4) and Index Tip (8)
            t_x, t_y = int(lm[4].x * w), int(lm[4].y * h)
            i_x, i_y = int(lm[8].x * w), int(lm[8].y * h)

            # Fast squared euclidean distance check (avoids sqrt)
            pinch_dist_sq = (t_x - i_x) ** 2 + (t_y - i_y) ** 2
            is_pinching = pinch_dist_sq < 1200  # ~35px threshold

            return (center_x, center_y, is_pinching, (t_x, t_y, i_x, i_y))
        return None
