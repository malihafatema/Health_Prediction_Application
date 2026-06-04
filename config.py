"""
config.py
=========
Centralised configuration for the Health Risk Predictor application.
All magic numbers, file paths and environment variable names live here
so the rest of the codebase never hard-codes strings or thresholds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present (never required in production — env vars take over)
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DB_PATH        = BASE_DIR / "database" / "patients.db"
MODEL_PATH     = BASE_DIR / "models"   / "model.pkl"

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Gemini Model ──────────────────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-1.5-flash"          # fast & cost-effective

# ── ML Feature Order (must match training) ───────────────────────────────────
ML_FEATURES    = ["glucose", "haemoglobin", "cholesterol"]

# ── Clinical Reference Ranges (used in UI hints & validation warnings) ────────
CLINICAL_RANGES = {
    "glucose": {
        "low":    (0,   70),
        "normal": (70,  99),
        "pre":    (100, 125),
        "high":   (126, 600),
        "unit":   "mg/dL",
    },
    "haemoglobin": {
        "low":    (0,   11.9),
        "normal": (12,  17),
        "high":   (17.1, 25),
        "unit":   "g/dL",
    },
    "cholesterol": {
        "desirable":  (0,   199),
        "borderline": (200, 239),
        "high":       (240, 600),
        "unit":       "mg/dL",
    },
}

# ── Risk Label Colours (Streamlit-compatible) ─────────────────────────────────
RISK_COLOURS = {
    "High Risk":   "#FF4B4B",
    "Medium Risk": "#FFA500",
    "Low Risk":    "#00C853",
}

# ── Pagination ────────────────────────────────────────────────────────────────
PATIENTS_PER_PAGE = 10

# ── App Meta ──────────────────────────────────────────────────────────────────
APP_TITLE   = "MIRA – Health Risk Predictor"
APP_ICON    = "🏥"
APP_VERSION = "1.0.0"
