"""
Trains a model on data/asl_landmarks.csv (produced by collect_data.py)
and saves it to model.pkl. main.py will automatically use this model
instead of the rule-based fallback if it exists.

Uses the engineered feature vector (hand_features.engineered_from_points)
on top of the raw normalized landmarks, and picks k via cross-validation
instead of guessing — both meaningfully improve precision over a plain
KNN-on-raw-coordinates baseline.

Usage:
    python train_model.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

from hand_features import engineered_from_points

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "asl_landmarks.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def build_features(raw_vectors):
    """raw_vectors: (N, 63) normalized-landmark rows from the CSV ->
    (N, 78) feature matrix (raw + engineered)."""
    feats = []
    for row in raw_vectors:
        pts = row.reshape(21, 3)
        eng = engineered_from_points(pts)
        feats.append(np.concatenate([row, eng]))
    return np.array(feats)


def pick_best_k(X, y, min_class_count):
    """Cross-validated search over k (and weighting) instead of a fixed guess."""
    max_k = max(1, min(9, min_class_count - 1))
    best_k, best_weights, best_score = 1, "distance", -1.0
    cv_folds = max(2, min(5, min_class_count))
    for k in range(1, max_k + 1):
        for weights in ("uniform", "distance"):
            clf = KNeighborsClassifier(n_neighbors=k, weights=weights)
            try:
                scores = cross_val_score(clf, X, y, cv=cv_folds)
            except ValueError:
                continue
            mean_score = scores.mean()
            if mean_score > best_score:
                best_k, best_weights, best_score = k, weights, mean_score
    return best_k, best_weights, best_score


def main():
    if not os.path.exists(DATA_PATH):
        print(f"No data found at {DATA_PATH}. Run collect_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    if len(df) < 20:
        print(f"Only {len(df)} samples found — collect more before training "
              f"(aim for 15+ per letter).")
        return

    raw = df.drop(columns=["label"]).values
    y = df["label"].values

    counts = pd.Series(y).value_counts()
    print("Samples per letter:")
    print(counts.to_string())
    if counts.min() < 8:
        print(f"\nNote: '{counts.idxmin()}' only has {counts.min()} samples — "
              f"more samples per letter (15-20+) will noticeably improve precision.")

    X = build_features(raw)

    best_k, best_weights, cv_score = pick_best_k(X, y, counts.min())
    print(f"\nBest k={best_k}, weights={best_weights} "
          f"(cross-val accuracy: {cv_score:.2%})")

    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    clf = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights)
    clf.fit(X_train, y_train)

    if len(X_test) > 0:
        preds = clf.predict(X_test)
        print("\nHeld-out evaluation:")
        print(classification_report(y_test, preds, zero_division=0))

    # refit on everything for the deployed model
    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model -> {MODEL_PATH}")
    print("main.py will pick this up automatically on next run.")


if __name__ == "__main__":
    main()
