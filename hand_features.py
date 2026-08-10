"""
Turns raw MediaPipe hand landmarks into two kinds of features:

1. A normalized flat vector (translated to the wrist, scaled by hand size,
   rotation-normalized) — used to train/run the ML classifier.
2. A small set of interpretable measurements (which fingers are extended,
   thumb position, etc.) — used by the rule-based fallback classifier.

Keeping these in one place means both classifiers see the same geometry.
"""

import math
import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "thumb":  (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index":  (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring":   (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky":  (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


def _to_xyz(landmarks):
    """landmarks: mediapipe NormalizedLandmarkList -> (21,3) numpy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)


def _angle_deg(a, b, c):
    """Angle at point b, formed by rays b->a and b->c, in degrees."""
    v1 = a - b
    v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_ang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_ang))


def normalized_vector(landmarks):
    """
    Rotation/scale/translation-normalized flat vector for ML use.
    - translate so wrist is origin
    - scale so wrist->middle_mcp distance = 1
    - rotate so that vector becomes the +y axis (hand "up" regardless of
      how it's held in frame)
    Returns a 63-length np.array (21 points x xyz).
    """
    pts = _to_xyz(landmarks)
    origin = pts[WRIST].copy()
    pts = pts - origin

    scale = np.linalg.norm(pts[MIDDLE_MCP][:2]) or 1e-6
    pts[:, :2] /= scale
    pts[:, 2] /= scale  # keep z in same relative scale

    ref = pts[MIDDLE_MCP][:2]
    theta = math.atan2(ref[0], ref[1] + 1e-9)  # angle needed to rotate ref onto +y
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    pts[:, :2] = pts[:, :2] @ rot.T

    return pts.flatten()


def finger_curl_angles(landmarks):
    """Dict of finger -> curl angle at the PIP/IP joint (180 = straight)."""
    pts = _to_xyz(landmarks)
    curls = {}
    for name, (mcp, pip_, dip, tip) in FINGERS.items():
        if name == "thumb":
            curls[name] = _angle_deg(pts[mcp], pts[pip_], pts[tip])
        else:
            curls[name] = _angle_deg(pts[mcp], pts[pip_], pts[tip])
    return curls


def hand_size(landmarks):
    pts = _to_xyz(landmarks)
    return np.linalg.norm(pts[WRIST][:2] - pts[MIDDLE_MCP][:2]) or 1e-6


def extension_flags(landmarks, straight_thresh=155.0):
    """Dict finger -> bool, True if extended (straight), based on curl angle."""
    curls = finger_curl_angles(landmarks)
    return {name: (angle >= straight_thresh) for name, angle in curls.items()}


def thumb_position(landmarks):
    """
    Classifies thumb position relative to the palm, which is the key signal
    for telling apart A / S / T / M / N / E and similar tucked-thumb letters.
    Returns one of: 'out' (extended away from palm), 'across' (tucked flat
    across the palm, tip near ring/pinky side), 'between_index_middle',
    'over_index' (on top of curled index, like 'T' / 'N').
    """
    pts = _to_xyz(landmarks)
    size = hand_size(landmarks)
    tip = pts[THUMB_TIP][:2]

    d_index_mcp = np.linalg.norm(tip - pts[INDEX_MCP][:2]) / size
    d_middle_mcp = np.linalg.norm(tip - pts[MIDDLE_MCP][:2]) / size
    d_ring_mcp = np.linalg.norm(tip - pts[RING_MCP][:2]) / size
    d_pinky_mcp = np.linalg.norm(tip - pts[PINKY_MCP][:2]) / size
    d_wrist = np.linalg.norm(tip - pts[WRIST][:2]) / size

    if d_wrist > 1.3 and d_pinky_mcp > 0.9:
        return "out"
    if d_index_mcp < 0.55:
        return "over_index"
    if d_middle_mcp < 0.6:
        return "between_index_middle"
    if d_ring_mcp < 0.7 or d_pinky_mcp < 0.9:
        return "across"
    return "out"


def feature_summary(landmarks):
    """Convenience bundle used by the rule-based classifier."""
    return {
        "extension": extension_flags(landmarks),
        "curls": finger_curl_angles(landmarks),
        "thumb_pos": thumb_position(landmarks),
    }


_TIP_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
_TIP_IDX = {"thumb": THUMB_TIP, "index": INDEX_TIP, "middle": MIDDLE_TIP,
            "ring": RING_TIP, "pinky": PINKY_TIP}
_PAIRS = [("thumb", "index"), ("index", "middle"), ("middle", "ring"),
          ("ring", "pinky"), ("thumb", "pinky")]


def engineered_from_points(pts):
    """
    Extra discriminative features computed from a (21,3) point array
    (works on either raw or normalized points — angles are rotation/scale
    invariant, and distances are meaningful as long as `pts` is already
    scale-normalized, which normalized_vector's output is).
    Returns a 15-length vector: 5 curl angles + 5 pairwise fingertip
    distances + 5 fingertip-to-wrist distances ("spread").
    """
    curls = np.array([
        _angle_deg(pts[FINGERS[n][0]], pts[FINGERS[n][1]], pts[FINGERS[n][3]])
        for n in _TIP_ORDER
    ]) / 180.0

    tip_pts = {n: pts[_TIP_IDX[n]][:2] for n in _TIP_ORDER}
    pair_dists = np.array([
        np.linalg.norm(tip_pts[a] - tip_pts[b]) for a, b in _PAIRS
    ])

    wrist = pts[WRIST][:2]
    spread = np.array([np.linalg.norm(tip_pts[n] - wrist) for n in _TIP_ORDER])

    return np.concatenate([curls, pair_dists, spread])


def full_feature_vector(landmarks):
    """
    The feature vector used for training/predicting with the ML model:
    normalized landmark positions (63) + engineered geometric features (15)
    = 78 dims. The engineered features give the classifier explicit access
    to the same curl/distance signals the rule-based classifier uses,
    which meaningfully improves precision over raw coordinates alone.
    """
    vec = normalized_vector(landmarks)
    pts = vec.reshape(21, 3)
    eng = engineered_from_points(pts)
    return np.concatenate([vec, eng])
