# 🏥 MIRA – Health Risk Predictor

> **Medical Intelligence & Risk Assessment** · Built for the GOKUL Infocare Junior AI/ML Developer Technical Assessment

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![Gemini](https://img.shields.io/badge/Google_Gemini-API-4285F4?logo=google)
![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Project Overview

MIRA is a full-stack, AI-powered health risk prediction application that helps healthcare professionals screen patients using three blood-test biomarkers: **Glucose**, **Haemoglobin**, and **Cholesterol**.

When a patient's data is submitted, a two-stage AI pipeline activates:

1. **Scikit-Learn RandomForestClassifier** classifies the patient as Low, Medium, or High Risk.
2. **Google Gemini API** generates a personalised 2-3 sentence clinical narrative stored in the Remarks field.

---

##  Features

Feature | Details |

CRUD Operations | Create, Read, Update, Delete patient records |
*Data Validation | Email format, future-date guard, numeric ranges, name sanitisation |
*ML Risk Classification| RandomForest trained on 3,000 clinically-grounded records |
Gemini AI Remarks | Personalised health narrative per patient via Gemini Flash |
Interactive Dashboard | KPI cards + 4 Plotly charts (donut, histogram, box, scatter) |
Search | Real-time search by name or email |
Export | Download all records as Excel (.xlsx) or CSV |
Graceful Degradation | Falls back to rule-based remarks if Gemini is unavailable |
Persistent Storage | SQLite with WAL mode for reliability |

---

##  Technology Stack

| Layer | Technology | Justification |
|---|---|---|
| Frontend | Streamlit | Rapid, Python-native UI; ideal for data apps |
| Backend | Python 3.11 | Clean ecosystem; first-class ML/AI library support |
| Database | SQLite | Zero-config, serverless, perfect for single-instance apps |
| ML | Scikit-Learn RandomForestClassifier | Handles non-linear interactions; robust to outliers |
| AI API | Google Gemini 1.5 Flash | Fast, cost-effective, excellent instruction-following |
| Charts | Plotly Express | Interactive, professional-grade visualisations |
| Export | Pandas + openpyxl | Industry standard for DataFrame → Excel pipelines |

---

##  Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)               │
│  Dashboard │ Add Patient │ Manage Patients │ Export   │
└──────────┬───────────────┬──────────────────┬─────────┘
           │               │                  │
    ┌──────▼──────┐  ┌─────▼──────┐   ┌──────▼──────┐
    │ validation  │  │ prediction │   │  database   │
    │    .py      │  │    .py     │   │    .py      │
    └─────────────┘  └─────┬──────┘   └─────────────┘
                           │                 │
                    ┌──────▼──────┐    ┌──────▼──────┐
                    │ ai_service  │    │  SQLite DB   │
                    │    .py      │    │ patients.db  │
                    └──────┬──────┘    └─────────────┘
                           │
                   ┌───────▼────────┐
                   │  Gemini Flash  │
                   │  (Google API)  │
                   └────────────────┘

AI/ML Pipeline:
  Patient Vitals → RandomForest → Risk Label → Gemini Prompt → Remarks
```

---

## 📁 Folder Structure

```
health-risk-predictor/
│
├── app.py               ← Streamlit entry point & all UI pages
├── database.py          ← CRUD data-access layer (SQLite)
├── prediction.py        ← ML model loading & predict_risk()
├── ai_service.py        ← Gemini API integration & fallback
├── validation.py        ← Pure-function input validation
├── config.py            ← Centralised constants & env vars
├── train_model.py       ← One-time model training script
│
├── models/
│   └── model.pkl        ← Trained RandomForestClassifier + LabelEncoder
│
├── database/
│   └── patients.db      ← SQLite database (auto-created on first run)
│
├── screenshots/         ← UI screenshots for documentation
│
├── requirements.txt     ← Python dependencies
├── .env.example         ← Environment variable template
├── .gitignore           ← Excludes secrets & build artefacts
└── README.md            ← This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- A Google Gemini API key (free tier available at [aistudio.google.com](https://aistudio.google.com))

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/health-risk-predictor.git
cd health-risk-predictor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Open .env and set GEMINI_API_KEY=your_key_here

# 5. Train the ML model (run once)
python train_model.py

# 6. Launch the application
streamlit run app.py
```

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

> **No Gemini key?** The app still works — a rule-based clinical remark is used as fallback. All other features are fully functional.

---

## 🤖 ML Model Explanation

### Why RandomForestClassifier?

| Reason | Detail |
|---|---|
| **Non-linear interactions** | High Glucose + Low Hb together carry more risk than either alone; Random Forest captures this |
| **No normalisation needed** | Unlike SVM/KNN, RF handles features on different scales (mg/dL vs g/dL) natively |
| **Feature importance** | Built-in `feature_importances_` enables clinical explainability |
| **Robust to outliers** | Medically abnormal values (very high glucose) don't destabilise the model |
| **Small data friendly** | Performs well on hundreds to thousands of samples |

### Features Used

| Feature | Why It Matters |
|---|---|
| **Glucose** | Elevated levels indicate diabetes risk (pre-diabetes ≥100, diabetes ≥126 mg/dL) |
| **Haemoglobin** | Low levels signal anaemia; excessively high may indicate polycythaemia |
| **Cholesterol** | High total cholesterol is a primary cardiovascular disease risk factor |

### Training Results

```
Dataset: 3,000 synthetic records (clinically grounded distributions)
Split  : 80% train / 20% test  (stratified)
Accuracy: 100% on held-out test set

Feature Importances:
  Glucose     : 36.5%
  Haemoglobin : 31.4%
  Cholesterol : 32.1%
```

> The high accuracy reflects the well-separated clinical ranges used in dataset generation — a real-world dataset (e.g. Pima Indians Diabetes) would achieve 78–85%, which is the expected performance.

---

## 🧠 Gemini API Integration

The AI pipeline follows this sequence:

```
1. Patient vitals + ML risk label assembled into a structured prompt
2. Prompt sent to gemini-1.5-flash (temperature=0.4 for clinical tone)
3. Response capped at 200 tokens → 2-3 concise sentences
4. Remark stored in the `remarks` column of the database
```

**Prompt design principles:**
- Role-assignment: "You are a clinical decision-support assistant"
- Structured output constraints: exactly 2-3 sentences
- Content guidance: concern + explanation + one actionable recommendation
- No disclaimers or bullet points (keeping it patient-friendly)

---

## 🗄️ Database Design

```sql
CREATE TABLE patients (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT    NOT NULL,
    dob         TEXT    NOT NULL,        -- ISO-8601: YYYY-MM-DD
    email       TEXT    NOT NULL UNIQUE,
    glucose     REAL    NOT NULL,
    haemoglobin REAL    NOT NULL,
    cholesterol REAL    NOT NULL,
    risk_level  TEXT,                    -- Low / Medium / High Risk
    remarks     TEXT,                    -- Gemini-generated narrative
    created_at  TEXT    NOT NULL,        -- ISO-8601 UTC datetime
    updated_at  TEXT    NOT NULL
);
```

---

## 🔮 Future Improvements

- [ ] **Authentication** — Role-based access (Doctor / Nurse / Admin)
- [ ] **Real Dataset Training** — Pima Indians Diabetes or NHANES dataset
- [ ] **More Biomarkers** — Blood pressure, BMI, HbA1c, eGFR
- [ ] **Trend Analysis** — Track patient markers over time with longitudinal charts
- [ ] **PDF Reports** — Auto-generate patient health report PDFs
- [ ] **REST API** — FastAPI backend to decouple frontend from business logic
- [ ] **Cloud Deployment** — Streamlit Community Cloud or Docker + AWS
- [ ] **FHIR Integration** — HL7 FHIR-compliant data exchange with EMR systems

---

## 📬 Contact

Built by Maliha Fatema as a technical assessment submission for GOKUL Infocare.

---

