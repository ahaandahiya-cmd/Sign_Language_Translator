"""
Calibration data collector.

Run this, hold up a letter's hand shape, press that letter key on your
keyboard to capture a burst of samples labeled with that letter. Repeat
for as many letters as you want the trained model to know (10-20 samples
per letter is plenty; more is better). Press ESC when done.

Samples are appended to data/asl_landmarks.csv. Run train_model.py
afterward to build the model.

Usage:
    python collect_data.py
"""

import csv
import os
import time

import cv2

from hand_features import normalized_vector
from mp_hand_detector import HandDetector, draw_landmarks

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "asl_landmarks.csv")
SAMPLES_PER_KEYPRESS = 12
VALID_LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY")  # J and Z need motion, skipped


def ensure_csv_header():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label"] + [f"f{i}" for i in range(63)])


def main():
    ensure_csv_header()
    detector = HandDetector(num_hands=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    capturing_until = 0
    capturing_label = None
    captured_count = 0
    total_this_session = 0

    print("Hold a hand shape, press its letter key (A-Y, no J/Z) to capture samples.")
    print("Press ESC to quit.")

    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = detector.detect(rgb)

            if hands:
                landmarks = hands[0]
                draw_landmarks(frame, landmarks)

                if time.time() < capturing_until:
                    vec = normalized_vector(landmarks)
                    writer.writerow([capturing_label] + vec.tolist())
                    captured_count += 1
                    total_this_session += 1
                    cv2.putText(frame, f"capturing {capturing_label}  {captured_count}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)

            cv2.putText(frame, f"session samples: {total_this_session}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
            cv2.putText(frame, "press a letter key to capture, ESC to quit",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("ASL calibration - collect data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key != 255 and chr(key).upper() in VALID_LETTERS:
                letter = chr(key).upper()
                capturing_label = letter
                captured_count = 0
                capturing_until = time.time() + (SAMPLES_PER_KEYPRESS / 20.0)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"Done. {total_this_session} samples added this session -> {DATA_PATH}")


if __name__ == "__main__":
    main()
