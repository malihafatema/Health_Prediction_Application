"""
database.py
===========
Data-access layer.  All SQL lives here; no other module touches the DB
directly.  Uses the standard-library sqlite3 module — no ORM overhead.

Schema
------
patients
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    full_name   TEXT    NOT NULL
    dob         TEXT    NOT NULL    (ISO-8601: YYYY-MM-DD)
    email       TEXT    NOT NULL    UNIQUE
    glucose     REAL    NOT NULL
    haemoglobin REAL    NOT NULL
    cholesterol REAL    NOT NULL
    risk_level  TEXT                (Low Risk / Medium Risk / High Risk)
    remarks     TEXT                (Gemini-generated narrative)
    created_at  TEXT    NOT NULL    (ISO-8601 datetime)
    updated_at  TEXT    NOT NULL    (ISO-8601 datetime)
"""

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import DB_PATH

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _get_connection() -> sqlite3.Connection:
    """Return a connection with row_factory set so rows behave like dicts."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # better concurrency
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ── Schema Initialisation ─────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they do not already exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS patients (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name   TEXT    NOT NULL,
        dob         TEXT    NOT NULL,
        email       TEXT    NOT NULL UNIQUE,
        glucose     REAL    NOT NULL,
        haemoglobin REAL    NOT NULL,
        cholesterol REAL    NOT NULL,
        risk_level  TEXT,
        remarks     TEXT,
        created_at  TEXT    NOT NULL,
        updated_at  TEXT    NOT NULL
    );
    """
    with _get_connection() as conn:
        conn.execute(ddl)
    logger.info("Database initialised at %s", DB_PATH)


# ── CREATE ────────────────────────────────────────────────────────────────────

def create_patient(
    full_name: str,
    dob: str,
    email: str,
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
    risk_level: str = "",
    remarks: str = "",
) -> int:
    """
    Insert a new patient record.
    Returns the new row id, or raises sqlite3.IntegrityError on duplicate email.
    """
    now = _now()
    sql = """
    INSERT INTO patients
        (full_name, dob, email, glucose, haemoglobin, cholesterol,
         risk_level, remarks, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_connection() as conn:
        cur = conn.execute(
            sql,
            (full_name, dob, email.lower().strip(),
             glucose, haemoglobin, cholesterol,
             risk_level, remarks, now, now),
        )
        patient_id = cur.lastrowid
    logger.info("Created patient id=%s  email=%s", patient_id, email)
    return patient_id


# ── READ ──────────────────────────────────────────────────────────────────────

def get_all_patients() -> list[dict]:
    """Return all patients ordered by creation date descending."""
    sql = "SELECT * FROM patients ORDER BY created_at DESC"
    with _get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def get_patient_by_id(patient_id: int) -> Optional[dict]:
    """Return a single patient dict or None."""
    sql = "SELECT * FROM patients WHERE id = ?"
    with _get_connection() as conn:
        row = conn.execute(sql, (patient_id,)).fetchone()
    return dict(row) if row else None


def search_patients(query: str) -> list[dict]:
    """Full-text search on name and email (case-insensitive)."""
    like = f"%{query.strip()}%"
    sql = """
    SELECT * FROM patients
    WHERE full_name LIKE ? OR email LIKE ?
    ORDER BY created_at DESC
    """
    with _get_connection() as conn:
        rows = conn.execute(sql, (like, like)).fetchall()
    return [dict(r) for r in rows]


def get_dashboard_stats() -> dict:
    """Return aggregate counts needed by the dashboard."""
    sql = """
    SELECT
        COUNT(*)                                        AS total,
        SUM(CASE WHEN risk_level = 'High Risk'   THEN 1 ELSE 0 END) AS high,
        SUM(CASE WHEN risk_level = 'Medium Risk' THEN 1 ELSE 0 END) AS medium,
        SUM(CASE WHEN risk_level = 'Low Risk'    THEN 1 ELSE 0 END) AS low
    FROM patients
    """
    with _get_connection() as conn:
        row = conn.execute(sql).fetchone()
    return dict(row)


# ── UPDATE ────────────────────────────────────────────────────────────────────

def update_patient(
    patient_id: int,
    full_name: str,
    dob: str,
    email: str,
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
    risk_level: str = "",
    remarks: str = "",
) -> bool:
    """
    Update all mutable fields of a patient record.
    Returns True if a row was affected.
    """
    sql = """
    UPDATE patients
    SET full_name   = ?,
        dob         = ?,
        email       = ?,
        glucose     = ?,
        haemoglobin = ?,
        cholesterol = ?,
        risk_level  = ?,
        remarks     = ?,
        updated_at  = ?
    WHERE id = ?
    """
    with _get_connection() as conn:
        cur = conn.execute(
            sql,
            (full_name, dob, email.lower().strip(),
             glucose, haemoglobin, cholesterol,
             risk_level, remarks, _now(), patient_id),
        )
        affected = cur.rowcount
    logger.info("Updated patient id=%s  rows_affected=%s", patient_id, affected)
    return affected > 0


# ── DELETE ────────────────────────────────────────────────────────────────────

def delete_patient(patient_id: int) -> bool:
    """Delete a patient by primary key.  Returns True if a row was removed."""
    sql = "DELETE FROM patients WHERE id = ?"
    with _get_connection() as conn:
        cur = conn.execute(sql, (patient_id,))
        affected = cur.rowcount
    logger.info("Deleted patient id=%s  rows_affected=%s", patient_id, affected)
    return affected > 0
