"""
Heuristic ASL fingerspelling classifier — no training data required.

v2: scores every candidate letter and picks the best match, rather than
the old cascading if/elif (which committed to the first partial match even
when a different letter fit the hand shape better — a real source of
wrong-letter errors). If no letter scores confidently, it returns None
instead of guessing.

Honest limitation up front: several ASL letters differ mainly by hand
*orientation* (P vs K, Q vs G) or by very fine thumb placement (M vs N vs
T vs S), which plain geometric rules struggle to separate reliably across
different hands/cameras. For those, calibrate on your own hand instead
(collect_data.py + train_model.py) — that's what actually fixes wrong-letter
errors long-term.

Supported here: A, B, C, D, E, F, I, L, M, N, O, S, T, U, V, W, X, Y
Deliberately not attempted by rules (too unreliable without calibration):
G, H, J, K, P, Q, R, Z
"""

import numpy as np
from hand_features import (
    feature_summary, THUMB_TIP, INDEX_TIP, MIDDLE_TIP,
    hand_size, _to_xyz,
)

MIN_CONFIDENCE = 0.62  # below this, we'd rather say "unknown" than guess wrong


def _dist(landmarks, i, j):
    pts = _to_xyz(landmarks)
    return np.linalg.norm(pts[i][:2] - pts[j][:2]) / hand_size(landmarks)


def _smooth(value, low, high):
    """0 at/below `low`, 1 at/above `high`, linear ramp between. Order-agnostic."""
    if low > high:
        low, high = high, low
        value = low + high - value  # mirror so ramp direction still works
    if high - low < 1e-6:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _finger_score(curls, expected):
    """expected: dict finger -> True(extended)/False(curled)/None(ignore).
    Returns mean continuous match score across the 4 non-thumb fingers."""
    scores = []
    for finger, want in expected.items():
        if want is None:
            continue
        angle = curls[finger]
        ext_score = _smooth(angle, 140, 172)  # 0=curled, 1=clearly straight
        scores.append(ext_score if want else 1 - ext_score)
    return float(np.mean(scores)) if scores else 1.0


def _thumb_state_score(curls, want):
    if want == "any":
        return 1.0
    angle = curls["thumb"]
    ext_score = _smooth(angle, 130, 165)
    return ext_score if want == "extended" else 1 - ext_score


def _thumb_pos_score(thumb_pos, allowed):
    if allowed is None:
        return 1.0
    return 1.0 if thumb_pos in allowed else 0.15


NON_THUMB = ("index", "middle", "ring", "pinky")


def _template_score(landmarks, curls, thumb_pos, fingers, thumb_state="any",
                     thumb_pos_ok=None, dist_checks=None):
    fscore = _finger_score(curls, {k: fingers.get(k) for k in NON_THUMB})
    tscore = _thumb_state_score(curls, thumb_state)
    pscore = _thumb_pos_score(thumb_pos, thumb_pos_ok)

    dscore = 1.0
    if dist_checks:
        parts = []
        for a, b, lo, hi in dist_checks:
            d = _dist(landmarks, a, b)
            # soft band: full score inside [lo,hi], falling off outside
            if lo <= d <= hi:
                parts.append(1.0)
            else:
                edge = lo if d < lo else hi
                parts.append(max(0.0, 1 - abs(d - edge) / 0.3))
        dscore = float(np.mean(parts))

    # weighted geometric mean so a weak factor pulls the score down hard
    weights = [fscore, tscore, pscore, dscore]
    return float(np.prod(weights)) ** (1 / len(weights))


def classify(landmarks):
    """Returns (letter_or_None, confidence 0-1)."""
    f = feature_summary(landmarks)
    curls = f["curls"]
    thumb_pos = f["thumb_pos"]

    thumb_index_d = _dist(landmarks, THUMB_TIP, INDEX_TIP)
    index_middle_d = _dist(landmarks, INDEX_TIP, MIDDLE_TIP)
    avg_curl_4 = np.mean([curls["index"], curls["middle"], curls["ring"], curls["pinky"]])

    candidates = {
        "B": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": True, "middle": True, "ring": True, "pinky": True},
        ),
        "W": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": True, "middle": True, "ring": True, "pinky": False},
        ),
        "U": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": True, "middle": True, "ring": False, "pinky": False},
            thumb_state="curled",
            dist_checks=[(INDEX_TIP, MIDDLE_TIP, 0.0, 0.35)],
        ),
        "V": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": True, "middle": True, "ring": False, "pinky": False},
            thumb_state="curled",
            dist_checks=[(INDEX_TIP, MIDDLE_TIP, 0.4, 1.2)],
        ),
        "Y": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": False, "middle": False, "ring": False, "pinky": True},
            thumb_state="extended",
        ),
        "L": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": True, "middle": False, "ring": False, "pinky": False},
            thumb_state="extended",
            thumb_pos_ok={"out"},
        ),
        "D": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": True, "middle": False, "ring": False, "pinky": False},
            thumb_state="curled",
        ),
        "I": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": False, "middle": False, "ring": False, "pinky": True},
            thumb_state="curled",
        ),
        "F": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": False, "middle": True, "ring": True, "pinky": True},
            thumb_state="curled",
            dist_checks=[(THUMB_TIP, INDEX_TIP, 0.0, 0.35)],
        ),
        "A": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": False, "middle": False, "ring": False, "pinky": False},
            thumb_pos_ok={"out"},
        ),
        "T": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": False, "middle": False, "ring": False, "pinky": False},
            thumb_pos_ok={"over_index"},
        ),
        "N": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": False, "middle": False, "ring": False, "pinky": False},
            thumb_pos_ok={"between_index_middle"},
        ),
        "M": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": False, "middle": False, "ring": False, "pinky": False},
            thumb_pos_ok={"across"},
        ) * _smooth(avg_curl_4, 100, 60),  # extra bonus for the tightest curl of the M/S/E group
        "S": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": False, "middle": False, "ring": False, "pinky": False},
            thumb_pos_ok={"across"},
        ) * (1 - abs(_smooth(avg_curl_4, 80, 130) - 0.5) * 1.2),
        "E": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": False, "middle": False, "ring": False, "pinky": False},
            thumb_pos_ok={"across"},
        ) * _smooth(avg_curl_4, 130, 160),
        "O": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": False, "middle": False, "ring": False, "pinky": False},
            thumb_pos_ok={"across", "over_index"},
            dist_checks=[(THUMB_TIP, INDEX_TIP, 0.0, 0.32)],
        ),
        "C": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": False, "middle": False, "ring": False, "pinky": False},
            thumb_pos_ok={"out"},
            dist_checks=[(THUMB_TIP, INDEX_TIP, 0.45, 0.9)],
        ) * _smooth(avg_curl_4, 90, 155) * (1 - _smooth(avg_curl_4, 155, 175)),
        "X": _template_score(
            landmarks, curls, thumb_pos,
            fingers={"index": None, "middle": False, "ring": False, "pinky": False},
            thumb_pos_ok={"across", "over_index"},
        ) * _smooth(curls["index"], 100, 150) * (1 - _smooth(curls["index"], 150, 172)),
    }

    best_letter = max(candidates, key=candidates.get)
    best_score = candidates[best_letter]

    if best_score < MIN_CONFIDENCE:
        return None, best_score

    # require a real margin over the runner-up, or we're likely picking
    # between two letters that both sort-of fit
    sorted_scores = sorted(candidates.values(), reverse=True)
    if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 0.06:
        return None, best_score

    return best_letter, best_score
