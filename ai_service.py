"""
ai_service.py
=============
Handles all communication with the Google Gemini API.

Flow:
    Patient vitals + ML risk label
        ↓
    Carefully engineered prompt
        ↓
    Gemini Flash  (fast, low-cost)
        ↓
    2-3 sentence clinical narrative stored as Remarks

Degrades gracefully: if the API key is missing or the call fails,
a rule-based fallback remark is returned so the app never crashes.
"""

import logging
import textwrap
from typing import Optional

import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

# ── Initialise SDK once ───────────────────────────────────────────────────────

_client_ready = False


def _ensure_client() -> None:
    global _client_ready
    if not _client_ready:
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set — AI remarks will use fallback.")
            return
        genai.configure(api_key=GEMINI_API_KEY)
        _client_ready = True


# ── Prompt Engineering ────────────────────────────────────────────────────────

def _build_prompt(
    full_name: str,
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
    risk_level: str,
    dob: str,
) -> str:
    return textwrap.dedent(f"""
    You are a clinical decision-support assistant.
    A healthcare screening system has processed the following patient data:

    Patient     : {full_name}
    Date of Birth: {dob}
    Glucose     : {glucose} mg/dL
    Haemoglobin : {haemoglobin} g/dL
    Cholesterol : {cholesterol} mg/dL
    ML Risk Assessment: {risk_level}

    Based on these values and the ML-predicted risk level, write a concise
    clinical remark of exactly 2-3 sentences that includes:
    1. The primary health concern suggested by these readings.
    2. A brief explanation of why this risk level was assigned.
    3. One actionable lifestyle recommendation for the patient.

    Write in plain English suitable for a patient-facing health record.
    Do NOT use bullet points or headers. Do NOT include disclaimers.
    """).strip()


# ── Fallback Remarks (rule-based) ─────────────────────────────────────────────

_FALLBACK = {
    "Low Risk": (
        "Your current blood markers are within healthy reference ranges, "
        "suggesting a low risk of metabolic or cardiovascular conditions. "
        "Continue maintaining a balanced diet, regular physical activity, "
        "and schedule annual health screenings."
    ),
    "Medium Risk": (
        "One or more of your blood markers fall in a borderline range, "
        "indicating a moderate risk that warrants attention. "
        "Consider consulting a healthcare professional, reducing processed-food "
        "intake, and increasing aerobic exercise to 150 minutes per week."
    ),
    "High Risk": (
        "Your readings indicate elevated levels that are associated with an "
        "increased risk of diabetes, anaemia, or cardiovascular disease. "
        "Please seek medical consultation promptly and follow a clinician-supervised "
        "diet and medication plan."
    ),
}


# ── Public API ────────────────────────────────────────────────────────────────

def generate_health_remark(
    full_name: str,
    dob: str,
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
    risk_level: str,
) -> str:
    """
    Generate a personalised health remark using the Gemini API.
    Falls back to a rule-based remark if the API is unavailable.

    Returns
    -------
    str : 2-3 sentence clinical narrative ready for storage in Remarks.
    """
    _ensure_client()

    if not GEMINI_API_KEY:
        logger.info("Using fallback remark (no API key).")
        return _FALLBACK.get(risk_level, _FALLBACK["Medium Risk"])

    prompt = _build_prompt(
        full_name, glucose, haemoglobin, cholesterol, risk_level, dob
    )

    try:
        model    = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,        # clinical tone — not too creative
                max_output_tokens=200,
            ),
        )
        remark = response.text.strip()
        logger.info("Gemini remark generated for '%s' (%s).", full_name, risk_level)
        return remark

    except Exception as exc:   # noqa: BLE001
        logger.error("Gemini API call failed: %s — using fallback.", exc)
        return _FALLBACK.get(risk_level, _FALLBACK["Medium Risk"])
