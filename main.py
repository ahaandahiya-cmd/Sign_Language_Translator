"""
Real-time ASL fingerspelling -> text.

Uses your trained model (model.pkl) if you've run collect_data.py +
train_model.py; otherwise falls back to the built-in rule-based classifier
automatically — works out of the box either way.

Controls:
    SPACE       insert a space
    BACKSPACE   delete last character
    C           clear the whole line
    S           save transcript to transcript.txt
    ESC / Q     quit

A letter is only typed once you hold that hand shape steady for a short
moment (HOLD_FRAMES) — this avoids spamming the same letter every frame
and roughly mirrors how you'd actually pause between letters.

Usage:
    python main.py
"""

import os
import pickle
import time
from collections import deque, Counter

import cv2
import numpy as np

from hand_features import full_feature_vector
from mp_hand_detector import HandDetector, draw_landmarks
import rule_classifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
TRANSCRIPT_PATH = os.path.join(os.path.dirname(__file__), "transcript.txt")

HOLD_FRAMES = 15          # consecutive frames of the same letter needed to commit it
COMMIT_FRACTION = 0.8     # fraction of those frames that must agree
MIN_MEAN_CONFIDENCE = 0.6 # average confidence over the held window
CONFIDENCE_THRESHOLD = 0.5
COOLDOWN_SEC = 0.6        # min time before the *same* letter can be typed again


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
        print(f"Loaded trained model from {MODEL_PATH}")
        return clf
    print("No trained model found — using built-in rule-based classifier. "
          "Run collect_data.py + train_model.py for better accuracy.")
    return None


def predict(landmarks, clf):
    if clf is not None:
        vec = full_feature_vector(landmarks).reshape(1, -1)
        pred = clf.predict(vec)[0]
        try:
            proba = clf.predict_proba(vec)[0]
            conf = float(np.max(proba))
        except Exception:
            conf = 1.0
        return pred, conf
    return rule_classifier.classify(landmarks)


def main():
    clf = load_model()
    detector = HandDetector(num_hands=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    text_buffer = ""
    recent_preds = deque(maxlen=HOLD_FRAMES)   # (letter_or_None, confidence)
    last_committed_letter = None
    last_commit_time = 0.0
    neutral_since_commit = True  # True once we've seen a "no clear sign" gap

    print("Running. Press ESC or Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands = detector.detect(rgb)
        current_letter, current_conf = None, 0.0

        if hands:
            landmarks = hands[0]
            draw_landmarks(frame, landmarks)
            letter, conf = predict(landmarks, clf)
            if letter and conf >= CONFIDENCE_THRESHOLD:
                current_letter, current_conf = letter, conf
                recent_preds.append((letter, conf))
            else:
                recent_preds.append((None, 0.0))
        else:
            recent_preds.append((None, 0.0))

        # a "neutral" moment (no confident sign) has to occur before the
        # same letter can be committed twice in a row — stops a slightly
        # long hold from being typed as a double letter
        if current_letter is None:
            neutral_since_commit = True

        # commit a letter once it's been the stable majority for HOLD_FRAMES,
        # with high enough average confidence over that window
        if len(recent_preds) == HOLD_FRAMES:
            letters_only = [p[0] for p in recent_preds]
            counts = Counter(letters_only)
            top_letter, top_count = counts.most_common(1)[0]
            now = time.time()

            if top_letter is not None:
                confs = [c for l, c in recent_preds if l == top_letter]
                mean_conf = sum(confs) / len(confs)
            else:
                mean_conf = 0.0

            is_repeat = (top_letter == last_committed_letter)
            allowed_repeat = (not is_repeat) or neutral_since_commit

            if (top_letter is not None
                    and top_count >= int(HOLD_FRAMES * COMMIT_FRACTION)
                    and mean_conf >= MIN_MEAN_CONFIDENCE
                    and allowed_repeat
                    and now - last_commit_time > COOLDOWN_SEC):
                text_buffer += top_letter
                last_committed_letter = top_letter
                last_commit_time = now
                neutral_since_commit = False
                recent_preds.clear()

        # ---- HUD ----
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (20, 15, 20), -1)
        cv2.rectangle(overlay, (0, h - 70), (w, h), (20, 15, 20), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        label = f"{current_letter}  ({current_conf:.2f})" if current_letter else "-"
        cv2.putText(frame, f"Detected: {label}", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 255, 190), 2)
        model_tag = "trained model" if clf is not None else "rule-based"
        cv2.putText(frame, model_tag, (16, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 200), 1)

        display_text = text_buffer[-60:] if len(text_buffer) > 60 else text_buffer
        cv2.putText(frame, display_text + "_", (16, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 220), 2)
        cv2.putText(frame, "SPACE space | BACKSPACE del | C clear | S save | Q quit",
                    (16, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

        cv2.imshow("ASL fingerspelling -> text", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key == 32:  # SPACE
            text_buffer += " "
        elif key in (8, 127):  # BACKSPACE
            text_buffer = text_buffer[:-1]
        elif key in (ord('c'), ord('C')):
            text_buffer = ""
        elif key in (ord('s'), ord('S')):
            with open(TRANSCRIPT_PATH, "a") as f:
                f.write(text_buffer + "\n")
            print(f"Saved to {TRANSCRIPT_PATH}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()