"""
Heuristic ASL fingerspelling classifier — no training data required.

Honest limitation up front: several ASL letters differ mainly by hand
*orientation* (P vs K, Q vs G) or by very fine thumb placement (M vs N vs T
vs S), which plain geometric rules struggle to separate reliably across
different hands/cameras. This module covers a solid, reliable subset.
For the rest (and for higher accuracy generally), use the calibration
pipeline (collect_data.py + train_model.py) which trains on YOUR hand.

Supported here: A, B, C, D, E, F, I, L, M, N, O, S, T, U, V, W, X, Y
Not attempted (orientation-dependent or too fine-grained for rules):
G, H, J, K, P, Q, R, Z
"""

import numpy as np
from hand_features import (
    feature_summary, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
    WRIST, hand_size, _to_xyz,
)


def _dist(landmarks, i, j):
    pts = _to_xyz(landmarks)
    return np.linalg.norm(pts[i][:2] - pts[j][:2]) / hand_size(landmarks)


def classify(landmarks):
    """Returns (letter_or_None, confidence 0-1)."""
    f = feature_summary(landmarks)
    ext = f["extension"]
    thumb_pos = f["thumb_pos"]
    curls = f["curls"]

    thumb, index, middle, ring, pinky = (
        ext["thumb"], ext["index"], ext["middle"], ext["ring"], ext["pinky"]
    )

    thumb_index_d = _dist(landmarks, THUMB_TIP, INDEX_TIP)
    index_middle_d = _dist(landmarks, INDEX_TIP, MIDDLE_TIP)

    # --- all four fingers extended ---
    if index and middle and ring and pinky:
        if thumb_pos == "out" or thumb:
            return "B", 0.85
        return "B", 0.6

    # --- three fingers (index, middle, ring) extended, pinky curled ---
    if index and middle and ring and not pinky:
        return "W", 0.75

    # --- two fingers extended: index + middle ---
    if index and middle and not ring and not pinky:
        if thumb_pos == "between_index_middle":
            return "R", 0.5  # crossed-finger detail not verified; low confidence
        if index_middle_d < 0.35:
            return "U", 0.75
        return "V", 0.75

    # --- thumb + pinky only ---
    if thumb and pinky and not index and not middle and not ring:
        return "Y", 0.85

    # --- thumb + index extended (L shape) ---
    if thumb and index and not middle and not ring and not pinky:
        return "L", 0.8

    # --- index only extended ---
    if index and not middle and not ring and not pinky and not thumb:
        if thumb_index_d < 0.4:
            return "D", 0.7
        return "D", 0.55

    # --- pinky only extended ---
    if pinky and not index and not middle and not ring and not thumb:
        return "I", 0.8

    # --- thumb + index pinched, middle/ring/pinky extended ---
    if middle and ring and pinky and not index:
        if thumb_index_d < 0.35:
            return "F", 0.7

    # --- all fingers curled (fist-family: A, S, T, M, N, E) ---
    if not index and not middle and not ring and not pinky:
        avg_curl = np.mean([curls["index"], curls["middle"], curls["ring"], curls["pinky"]])

        if thumb_pos == "out":
            return "A", 0.75
        if thumb_pos == "over_index":
            return "T", 0.55
        if thumb_pos == "between_index_middle":
            return "N", 0.55
        if thumb_pos == "across":
            if thumb_index_d < 0.3:
                return "O", 0.6
            if avg_curl < 80:
                return "M", 0.5
            if avg_curl < 130:
                return "S", 0.65
            return "E", 0.55

    # --- moderately curved fingers, thumb out (C shape) ---
    avg_curl_all = np.mean(list(curls.values()))
    if 90 <= avg_curl_all <= 160 and thumb_pos == "out" and 0.4 < thumb_index_d < 0.9:
        return "C", 0.55

    # --- thumb crosses palm, all fingers curled tight, spread wide (X-ish) ---
    if index and not middle and not ring and not pinky and curls["index"] < 155:
        return "X", 0.4

    return None, 0.0
