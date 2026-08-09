"""
Wraps MediaPipe's HandLandmarker (Tasks API) so main.py / collect_data.py
don't duplicate setup. Handles downloading the model file on first run.

Note: newer mediapipe releases (0.10.2x+) removed the old
`mediapipe.solutions.hands` API entirely — this uses the current
supported API instead.
"""

import os
import time
import urllib.request

import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import mediapipe as mp

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Standard 21-point hand skeleton connections, for manual drawing
# (the old mp.solutions.drawing_utils helper no longer exists).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand-tracking model (one-time, ~10MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


class HandDetector:
    def __init__(self, num_hands=1):
        ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def detect(self, frame_rgb):
        """frame_rgb: HxWx3 uint8 numpy array (RGB). Returns list of hand
        landmark lists (each a list of 21 objects with .x/.y/.z), or []."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.hand_landmarks or []

    def close(self):
        self.landmarker.close()


def draw_landmarks(frame_bgr, landmarks):
    """Draws the skeleton for one hand's landmarks onto a BGR frame in place."""
    import cv2
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (120, 200, 255), 2)
    for x, y in pts:
        cv2.circle(frame_bgr, (x, y), 4, (255, 140, 180), -1)
