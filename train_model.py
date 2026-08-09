"""
Trains a model on data/asl_landmarks.csv (produced by collect_data.py)
and saves it to model.pkl. main.py will automatically use this model
instead of the rule-based fallback if it exists.

Usage:
    python train_model.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "asl_landmarks.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"No data found at {DATA_PATH}. Run collect_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    if len(df) < 20:
        print(f"Only {len(df)} samples found — collect more before training "
              f"(aim for 10+ per letter).")
        return

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    counts = pd.Series(y).value_counts()
    print("Samples per letter:")
    print(counts.to_string())

    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    k = min(5, max(1, len(X_train) // 10))
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    clf.fit(X_train, y_train)

    if len(X_test) > 0:
        preds = clf.predict(X_test)
        print("\nHeld-out evaluation:")
        print(classification_report(y_test, preds, zero_division=0))

    # refit on everything for the deployed model
    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved model -> {MODEL_PATH}")
    print("main.py will pick this up automatically on next run.")


if __name__ == "__main__":
    main()
