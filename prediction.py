"""
prediction.py
=============
Loads the pre-trained RandomForestClassifier and exposes a single
predict() function that accepts raw blood-test values and returns a
risk-level label.

The model artefact (models/model.pkl) is loaded once at module import
time so it is not re-read on every prediction call.
"""

import pickle
import logging
from pathlib import Path

import numpy as np

from config import MODEL_PATH

logger = logging.getLogger(__name__)

# ── Load model once at startup ────────────────────────────────────────────────

_MODEL_BUNDLE: dict | None = None


def _load_model() -> dict:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"ML model not found at '{MODEL_PATH}'. "
                "Please run  python train_model.py  first."
            )
        with open(MODEL_PATH, "rb") as f:
            _MODEL_BUNDLE = pickle.load(f)
        logger.info("ML model loaded from %s", MODEL_PATH)
    return _MODEL_BUNDLE


# ── Public API ────────────────────────────────────────────────────────────────

def predict_risk(glucose: float, haemoglobin: float, cholesterol: float) -> str:
    """
    Run the RandomForestClassifier on the three blood-test features.

    Parameters
    ----------
    glucose      : blood glucose level   (mg/dL)
    haemoglobin  : haemoglobin level     (g/dL)
    cholesterol  : total cholesterol     (mg/dL)

    Returns
    -------
    One of:  "Low Risk"  |  "Medium Risk"  |  "High Risk"

    Why RandomForest?
    -----------------
    • Handles non-linear feature interactions (e.g. high glucose AND low Hb
      together are more dangerous than either alone).
    • Robust to outliers without needing normalisation.
    • Built-in feature-importance scores aid clinical explainability.
    • Excellent performance on tabular data with few features.
    """
    bundle = _load_model()
    model  = bundle["model"]
    le     = bundle["label_encoder"]

    X        = np.array([[glucose, haemoglobin, cholesterol]], dtype=float)
    pred_idx = model.predict(X)[0]
    label    = le.inverse_transform([pred_idx])[0]

    logger.debug(
        "Prediction: glucose=%.1f  hb=%.1f  chol=%.1f  → %s",
        glucose, haemoglobin, cholesterol, label,
    )
    return label


def get_risk_probabilities(
    glucose: float, haemoglobin: float, cholesterol: float
) -> dict[str, float]:
    """
    Return the per-class probability scores from the model.
    Useful for displaying confidence in the UI.
    """
    bundle = _load_model()
    model  = bundle["model"]
    le     = bundle["label_encoder"]

    X     = np.array([[glucose, haemoglobin, cholesterol]], dtype=float)
    probs = model.predict_proba(X)[0]
    return {label: round(float(p), 4)
            for label, p in zip(le.classes_, probs)}
