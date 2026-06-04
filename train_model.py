"""
train_model.py
==============
Trains a RandomForestClassifier on a synthetic but medically-grounded
dataset derived from standard clinical thresholds for Glucose, Haemoglobin
and Cholesterol.  Saves the fitted model and the label encoder to
models/model.pkl so the main application never re-trains at runtime.

Run once:  python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── 1. Synthetic Dataset Generation ─────────────────────────────────────────
#
# Clinical reference ranges used to build realistic distributions:
#
#  Glucose   (mg/dL):  Normal <100  Pre-diabetic 100-125  Diabetic ≥126
#  Haemoglobin (g/dL): Low <12 (F) / <13.5 (M)   Normal 12-17   High >17
#  Cholesterol (mg/dL):Desirable <200  Borderline 200-239  High ≥240
#
# Risk label mapping:
#  Low Risk    – all markers in healthy range
#  Medium Risk – one or two markers borderline
#  High Risk   – one or more markers in clinical alert range

def generate_dataset(n_samples: int = 3000) -> pd.DataFrame:
    records = []

    # --- LOW RISK (40 % of data) ---
    n_low = int(n_samples * 0.40)
    glucose_low      = np.random.normal(90,  10, n_low).clip(60, 99)
    haemoglobin_low  = np.random.normal(14,  1,  n_low).clip(12, 17)
    cholesterol_low  = np.random.normal(175, 15, n_low).clip(100, 199)
    for g, h, c in zip(glucose_low, haemoglobin_low, cholesterol_low):
        records.append({"glucose": round(g, 1),
                         "haemoglobin": round(h, 1),
                         "cholesterol": round(c, 1),
                         "risk": "Low Risk"})

    # --- MEDIUM RISK (35 % of data) ---
    n_med = int(n_samples * 0.35)
    glucose_med      = np.random.normal(112, 8,  n_med).clip(100, 125)
    haemoglobin_med  = np.random.normal(11,  1,  n_med).clip(9, 12)
    cholesterol_med  = np.random.normal(220, 10, n_med).clip(200, 239)
    for g, h, c in zip(glucose_med, haemoglobin_med, cholesterol_med):
        records.append({"glucose": round(g, 1),
                         "haemoglobin": round(h, 1),
                         "cholesterol": round(c, 1),
                         "risk": "Medium Risk"})

    # --- HIGH RISK (25 % of data) ---
    n_high = n_samples - n_low - n_med
    glucose_high      = np.random.normal(155, 20, n_high).clip(126, 300)
    haemoglobin_high  = np.random.normal(8,   1,  n_high).clip(5, 11)
    cholesterol_high  = np.random.normal(260, 20, n_high).clip(240, 400)
    for g, h, c in zip(glucose_high, haemoglobin_high, cholesterol_high):
        records.append({"glucose": round(g, 1),
                         "haemoglobin": round(h, 1),
                         "cholesterol": round(c, 1),
                         "risk": "High Risk"})

    df = pd.DataFrame(records).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


# ── 2. Train ─────────────────────────────────────────────────────────────────
def train():
    print("▶ Generating synthetic clinical dataset …")
    df = generate_dataset(3000)
    print(f"  Dataset shape : {df.shape}")
    print(f"  Label counts  :\n{df['risk'].value_counts()}\n")

    X = df[["glucose", "haemoglobin", "cholesterol"]].values
    le = LabelEncoder()
    y  = le.fit_transform(df["risk"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("▶ Training RandomForestClassifier …")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n✅ Accuracy on held-out test set : {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    features = ["Glucose", "Haemoglobin", "Cholesterol"]
    importances = clf.feature_importances_
    print("\nFeature Importances:")
    for feat, imp in zip(features, importances):
        print(f"  {feat}: {imp:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump({"model": clf, "label_encoder": le}, f)

    print("\n✅ Model saved to  models/model.pkl")


if __name__ == "__main__":
    train()
