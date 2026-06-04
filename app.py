"""
app.py
======
Entry point for the MIRA Health Risk Predictor Streamlit application.

Navigation pages
----------------
  🏠 Dashboard        – KPI cards + interactive charts
  ➕ Add Patient       – validated form → ML → Gemini → DB
  📋 Manage Patients   – search, view, update, delete
  📤 Export Data       – download as Excel

Run with:
    streamlit run app.py
"""

import logging
import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import database as db
from ai_service import generate_health_remark
from config import APP_ICON, APP_TITLE, RISK_COLOURS
from prediction import get_risk_probabilities, predict_risk
from validation import validate_patient_form

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    section[data-testid="stSidebar"] * { color: #e8f4f8 !important; }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        padding: 6px 0;
    }

    /* ── Metric cards ────────────────────────────────────── */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e8ecf0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all .2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,.15); }

    /* ── Risk badges ─────────────────────────────────────── */
    .badge-high   { background:#ffe5e5; color:#c0392b; border-radius:20px; padding:3px 12px; font-size:.82rem; font-weight:600; }
    .badge-medium { background:#fff3e0; color:#e67e22; border-radius:20px; padding:3px 12px; font-size:.82rem; font-weight:600; }
    .badge-low    { background:#e8f5e9; color:#27ae60; border-radius:20px; padding:3px 12px; font-size:.82rem; font-weight:600; }

    /* ── Section dividers ────────────────────────────────── */
    hr { border: 1px solid #e8ecf0; margin: 1.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── DB init ───────────────────────────────────────────────────────────────────
db.init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _risk_badge(risk: str) -> str:
    cls = {"High Risk": "high", "Medium Risk": "medium", "Low Risk": "low"}.get(risk, "low")
    return f'<span class="badge-{cls}">{risk}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def page_dashboard() -> None:
    st.title("🏥 MIRA – Health Risk Dashboard")
    st.markdown("*Medical Intelligence & Risk Assessment Platform*")
    st.markdown("---")

    stats = db.get_dashboard_stats()
    patients = db.get_all_patients()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👤 Total Patients",  stats["total"]  or 0)
    c2.metric("🔴 High Risk",        stats["high"]   or 0)
    c3.metric("🟠 Medium Risk",      stats["medium"] or 0)
    c4.metric("🟢 Low Risk",         stats["low"]    or 0)

    if not patients:
        st.info("No patient data yet. Add patients to see analytics here.")
        return

    df = pd.DataFrame(patients)

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ── Risk Distribution Donut ───────────────────────────────────────────────
    with col1:
        st.subheader("📊 Risk Distribution")
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            hole=0.45,
            color="Risk Level",
            color_discrete_map=RISK_COLOURS,
        )
        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Glucose Distribution Histogram ───────────────────────────────────────
    with col2:
        st.subheader("🩸 Glucose Distribution")
        fig2 = px.histogram(
            df, x="glucose", color="risk_level",
            color_discrete_map=RISK_COLOURS,
            nbins=20, barmode="overlay",
            labels={"glucose": "Glucose (mg/dL)", "risk_level": "Risk"},
            opacity=0.75,
        )
        fig2.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # ── Cholesterol Box Plot ──────────────────────────────────────────────────
    with col3:
        st.subheader("💊 Cholesterol by Risk Level")
        fig3 = px.box(
            df, x="risk_level", y="cholesterol",
            color="risk_level", color_discrete_map=RISK_COLOURS,
            labels={"cholesterol": "Cholesterol (mg/dL)", "risk_level": "Risk Level"},
        )
        fig3.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Haemoglobin Scatter ───────────────────────────────────────────────────
    with col4:
        st.subheader("🔬 Glucose vs Haemoglobin")
        fig4 = px.scatter(
            df, x="glucose", y="haemoglobin",
            color="risk_level", color_discrete_map=RISK_COLOURS,
            hover_data=["full_name", "cholesterol"],
            labels={
                "glucose": "Glucose (mg/dL)",
                "haemoglobin": "Haemoglobin (g/dL)",
                "risk_level": "Risk",
            },
        )
        fig4.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Recent Patients Table ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🕐 Recently Added Patients")
    recent = df.head(5)[["full_name", "email", "glucose", "haemoglobin", "cholesterol", "risk_level", "created_at"]]
    recent.columns = ["Name", "Email", "Glucose", "Haemoglobin", "Cholesterol", "Risk", "Added"]
    st.dataframe(recent, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Add Patient
# ─────────────────────────────────────────────────────────────────────────────

def page_add_patient() -> None:
    st.title("➕ Add New Patient")
    st.markdown("Complete all fields. AI risk analysis runs automatically on submission.")
    st.markdown("---")

    with st.form("add_patient_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            full_name   = st.text_input("👤 Full Name *",         placeholder="e.g. Priya Sharma")
            email       = st.text_input("✉️ Email Address *",      placeholder="e.g. priya@example.com")
            glucose     = st.number_input("🩸 Glucose (mg/dL) *",   min_value=0.0, max_value=600.0, value=0.0, step=0.1, format="%.1f")
        with c2:
            dob         = st.date_input("📅 Date of Birth *")
            haemoglobin = st.number_input("🔬 Haemoglobin (g/dL) *", min_value=0.0, max_value=25.0,  value=0.0, step=0.1, format="%.1f")
            cholesterol = st.number_input("💊 Cholesterol (mg/dL) *", min_value=0.0, max_value=600.0, value=0.0, step=0.1, format="%.1f")

        st.markdown(
            "<small>Reference: Glucose Normal &lt;100 | Haemoglobin Normal 12–17 | Cholesterol Desirable &lt;200</small>",
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button("🚀 Analyse & Save Patient", use_container_width=True)

    if not submitted:
        return

    # ── Validation ────────────────────────────────────────────────────────────
    errors = validate_patient_form(
        full_name, str(dob), email, glucose, haemoglobin, cholesterol
    )
    if errors:
        for msg in errors.values():
            st.error(f"❌ {msg}")
        return

    # ── ML Prediction ─────────────────────────────────────────────────────────
    with st.spinner("🤖 Running ML risk assessment…"):
        risk_level = predict_risk(glucose, haemoglobin, cholesterol)
        probs      = get_risk_probabilities(glucose, haemoglobin, cholesterol)

    # ── Gemini Remark ─────────────────────────────────────────────────────────
    with st.spinner("✨ Generating AI health remark via Gemini…"):
        remark = generate_health_remark(
            full_name, str(dob), glucose, haemoglobin, cholesterol, risk_level
        )

    # ── Persist ───────────────────────────────────────────────────────────────
    try:
        patient_id = db.create_patient(
            full_name, str(dob), email, glucose, haemoglobin, cholesterol,
            risk_level, remark,
        )
    except sqlite3.IntegrityError:
        st.error("❌ A patient with this email address already exists.")
        return

    # ── Result Card ───────────────────────────────────────────────────────────
    colour = RISK_COLOURS.get(risk_level, "#888")
    st.success(f"✅ Patient #{patient_id} saved successfully!")

    st.markdown(
        f"""
        <div style="border:2px solid {colour}; border-radius:12px; padding:20px; margin-top:16px;">
            <h3 style="color:{colour}; margin:0 0 8px 0;">⚕️ Risk Assessment: {risk_level}</h3>
            <p style="margin:0; color:#444; line-height:1.7;">{remark}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability bars
    st.markdown("#### Model Confidence")
    for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
        st.progress(prob, text=f"{label}: {prob*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Manage Patients
# ─────────────────────────────────────────────────────────────────────────────

def page_manage_patients() -> None:
    st.title("📋 Manage Patients")
    st.markdown("---")

    # ── Search ────────────────────────────────────────────────────────────────
    query = st.text_input("🔍 Search by name or email", placeholder="Type to search…")
    patients = db.search_patients(query) if query else db.get_all_patients()

    if not patients:
        st.info("No patients found.")
        return

    st.markdown(f"**{len(patients)} record(s) found**")

    # ── Patient Cards ─────────────────────────────────────────────────────────
    for p in patients:
        with st.expander(
            f"#{p['id']}  {p['full_name']}  —  {_risk_badge(p['risk_level'])}",
            expanded=False,
        ):
            col_a, col_b = st.columns([3, 1])

            with col_a:
                st.markdown(
                    f"""
                    | Field | Value |
                    |---|---|
                    | **Email** | {p['email']} |
                    | **Date of Birth** | {p['dob']} |
                    | **Glucose** | {p['glucose']} mg/dL |
                    | **Haemoglobin** | {p['haemoglobin']} g/dL |
                    | **Cholesterol** | {p['cholesterol']} mg/dL |
                    | **Added** | {p['created_at']} |
                    """
                )
                if p["remarks"]:
                    st.info(f"💬 **AI Remarks:** {p['remarks']}")

            with col_b:
                # ── Edit ──────────────────────────────────────────────────────
                if st.button("✏️ Edit", key=f"edit_{p['id']}"):
                    st.session_state["edit_id"] = p["id"]
                    st.rerun()

                # ── Delete ────────────────────────────────────────────────────
                if st.button("🗑️ Delete", key=f"del_{p['id']}"):
                    st.session_state["confirm_delete"] = p["id"]
                    st.rerun()

                if st.session_state.get("confirm_delete") == p["id"]:
                    st.warning("Are you sure?")
                    if st.button("✅ Confirm Delete", key=f"conf_{p['id']}"):
                        db.delete_patient(p["id"])
                        st.session_state.pop("confirm_delete", None)
                        st.success("Patient deleted.")
                        st.rerun()

    # ── Edit Modal ────────────────────────────────────────────────────────────
    edit_id = st.session_state.get("edit_id")
    if edit_id:
        patient = db.get_patient_by_id(edit_id)
        if patient:
            st.markdown("---")
            st.subheader(f"✏️ Editing: {patient['full_name']}")
            with st.form("edit_form"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    e_name  = st.text_input("Full Name",    value=patient["full_name"])
                    e_email = st.text_input("Email",        value=patient["email"])
                    e_gluc  = st.number_input("Glucose",    value=float(patient["glucose"]), step=0.1, format="%.1f")
                with ec2:
                    from datetime import date as _date
                    e_dob   = st.date_input("Date of Birth", value=_date.fromisoformat(patient["dob"]))
                    e_hb    = st.number_input("Haemoglobin", value=float(patient["haemoglobin"]), step=0.1, format="%.1f")
                    e_chol  = st.number_input("Cholesterol", value=float(patient["cholesterol"]), step=0.1, format="%.1f")

                reanalyse = st.checkbox("♻️ Re-run AI/ML analysis", value=True)
                save_btn  = st.form_submit_button("💾 Save Changes", use_container_width=True)
                cancel    = st.form_submit_button("Cancel")

            if cancel:
                st.session_state.pop("edit_id", None)
                st.rerun()

            if save_btn:
                errors = validate_patient_form(e_name, str(e_dob), e_email, e_gluc, e_hb, e_chol)
                if errors:
                    for msg in errors.values():
                        st.error(f"❌ {msg}")
                else:
                    risk   = patient["risk_level"]
                    remark = patient["remarks"]
                    if reanalyse:
                        risk   = predict_risk(e_gluc, e_hb, e_chol)
                        remark = generate_health_remark(e_name, str(e_dob), e_gluc, e_hb, e_chol, risk)
                    db.update_patient(edit_id, e_name, str(e_dob), e_email, e_gluc, e_hb, e_chol, risk, remark)
                    st.session_state.pop("edit_id", None)
                    st.success("✅ Patient updated.")
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Page: Export
# ─────────────────────────────────────────────────────────────────────────────

def page_export() -> None:
    st.title("📤 Export Patient Data")
    st.markdown("---")

    patients = db.get_all_patients()
    if not patients:
        st.info("No data to export yet.")
        return

    df = pd.DataFrame(patients)
    df.columns = [c.replace("_", " ").title() for c in df.columns]

    st.markdown(f"**{len(df)} patient record(s) ready for export.**")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Excel ──────────────────────────────────────────────────────────────────
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Patients")
        # Summary sheet
        stats = db.get_dashboard_stats()
        summary_df = pd.DataFrame(
            [{"Metric": k.title(), "Value": v} for k, v in stats.items()]
        )
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

    st.download_button(
        label="⬇️ Download as Excel (.xlsx)",
        data=buf.getvalue(),
        file_name="mira_patients_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ── CSV ────────────────────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download as CSV (.csv)",
        data=df.to_csv(index=False),
        file_name="mira_patients_export.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 12px 0 20px 0;">
                <div style="font-size:2.5rem;">🏥</div>
                <div style="font-size:1.2rem; font-weight:700; letter-spacing:.5px;">MIRA</div>
                <div style="font-size:.75rem; opacity:.7;">Health Risk Predictor</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            options=[
                "🏠 Dashboard",
                "➕ Add Patient",
                "📋 Manage Patients",
                "📤 Export Data",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        stats = db.get_dashboard_stats()
        st.markdown(
            f"""
            <div style="font-size:.8rem; opacity:.8; line-height:1.8;">
            👤 Total: <b>{stats['total'] or 0}</b><br>
            🔴 High: <b>{stats['high'] or 0}</b><br>
            🟠 Medium: <b>{stats['medium'] or 0}</b><br>
            🟢 Low: <b>{stats['low'] or 0}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Route ──────────────────────────────────────────────────────────────────
    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "➕ Add Patient":
        page_add_patient()
    elif page == "📋 Manage Patients":
        page_manage_patients()
    elif page == "📤 Export Data":
        page_export()


if __name__ == "__main__":
    main()
