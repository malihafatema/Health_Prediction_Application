"""
validation.py
=============
Pure-function validation layer.  Every function returns a (bool, str) tuple:
    (True,  "")           – input is valid
    (False, "error msg")  – input failed validation
No side-effects; nothing imported from the rest of the project.
"""

import re
from datetime import date, datetime
from typing import Optional


# ── Email ─────────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


def validate_email(email: str) -> tuple[bool, str]:
    email = email.strip()
    if not email:
        return False, "Email address is required."
    if not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address (e.g. john@example.com)."
    return True, ""


# ── Date of Birth ────────────────────────────────────────────────────────────

def validate_dob(dob: str) -> tuple[bool, str]:
    """
    Accepts ISO-8601 date strings (YYYY-MM-DD).
    Rules:
      • Must be parseable
      • Cannot be a future date
      • Cannot be unrealistically old (> 130 years ago)
    """
    if not dob:
        return False, "Date of birth is required."
    try:
        dob_date = datetime.strptime(str(dob), "%Y-%m-%d").date()
    except ValueError:
        return False, "Date of birth must be in YYYY-MM-DD format."

    today = date.today()
    if dob_date > today:
        return False, "Date of birth cannot be a future date."
    if (today - dob_date).days > 130 * 365:
        return False, "Date of birth seems unrealistic (more than 130 years ago)."
    return True, ""


# ── Numeric Blood-Test Values ─────────────────────────────────────────────────

def _validate_positive_number(
    value,
    field_name: str,
    min_val: float,
    max_val: float,
) -> tuple[bool, str]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False, f"{field_name} must be a numeric value."
    if num <= 0:
        return False, f"{field_name} must be greater than zero."
    if num < min_val or num > max_val:
        return (
            False,
            f"{field_name} must be between {min_val} and {max_val}."
        )
    return True, ""


def validate_glucose(value) -> tuple[bool, str]:
    return _validate_positive_number(value, "Glucose", 20, 600)


def validate_haemoglobin(value) -> tuple[bool, str]:
    return _validate_positive_number(value, "Haemoglobin", 1, 25)


def validate_cholesterol(value) -> tuple[bool, str]:
    return _validate_positive_number(value, "Cholesterol", 50, 600)


# ── Full Name ─────────────────────────────────────────────────────────────────

def validate_full_name(name: str) -> tuple[bool, str]:
    name = name.strip()
    if not name:
        return False, "Full name is required."
    if len(name) < 2:
        return False, "Full name must be at least 2 characters."
    if len(name) > 120:
        return False, "Full name must not exceed 120 characters."
    if re.search(r"[0-9]", name):
        return False, "Full name must not contain numbers."
    return True, ""


# ── Composite Patient Validator ───────────────────────────────────────────────

def validate_patient_form(
    full_name: str,
    dob: str,
    email: str,
    glucose,
    haemoglobin,
    cholesterol,
) -> dict[str, str]:
    """
    Validate all patient fields at once.
    Returns a dict of {field: error_message} — empty dict means all valid.
    """
    errors: dict[str, str] = {}

    checks = [
        ("full_name",   validate_full_name(full_name)),
        ("dob",         validate_dob(dob)),
        ("email",       validate_email(email)),
        ("glucose",     validate_glucose(glucose)),
        ("haemoglobin", validate_haemoglobin(haemoglobin)),
        ("cholesterol", validate_cholesterol(cholesterol)),
    ]

    for field, (ok, msg) in checks:
        if not ok:
            errors[field] = msg

    return errors
