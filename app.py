import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CreditWise", page_icon="💳", layout="centered")

st.title("💳 CreditWise — Loan Approval Predictor")
st.markdown("Fill in the applicant details below and get an instant prediction.")
st.divider()

# ── Load model files ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model   = joblib.load("loan_model.pkl")
    scaler  = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_models()
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}. Make sure loan_model.pkl, scaler.pkl, encoder.pkl are in the same folder as app.py.")
    st.stop()

CATEGORICAL_COLS = ["Employment_Status", "Employer_Category", "Loan_Purpose",
                    "Property_Area", "Gender", "Marital_Status"]

# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("predict_form"):

    st.subheader("👤 Personal Details")
    c1, c2, c3 = st.columns(3)
    age            = c1.number_input("Age", min_value=18, max_value=80, value=35)
    gender         = c2.selectbox("Gender", ["Male", "Female"])
    marital_status = c3.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    c4, c5 = st.columns(2)
    education    = c4.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    employment   = c5.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed"])
    employer_cat = st.selectbox("Employer Category", ["Government", "Private", "NGO", "Other"])

    st.subheader("💰 Financial Details")
    c6, c7 = st.columns(2)
    applicant_income   = c6.number_input("Applicant Income (₹)", min_value=0, max_value=1_000_000, value=50_000, step=1000)
    coapplicant_income = c7.number_input("Co-applicant Income (₹)", min_value=0, max_value=500_000, value=0, step=1000)

    c8, c9 = st.columns(2)
    loan_amount  = c8.number_input("Loan Amount (₹)", min_value=10_000, max_value=5_000_000, value=200_000, step=10000)
    savings      = c9.number_input("Savings (₹)", min_value=0, max_value=2_000_000, value=50_000, step=1000)

    c10, c11 = st.columns(2)
    credit_score = c10.slider("Credit Score", min_value=300, max_value=900, value=700)
    dti_ratio    = c11.slider("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

    st.subheader("🏠 Loan Details")
    c12, c13 = st.columns(2)
    loan_purpose  = c12.selectbox("Loan Purpose", ["Home", "Education", "Medical", "Personal", "Business"])
    property_area = c13.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submitted = st.form_submit_button("🔮 Predict Loan Approval", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # Build input dataframe
    edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
    input_df = pd.DataFrame([{
        "Age":                age,
        "Applicant_Income":   applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Loan_Amount":        loan_amount,
        "Credit_Score":       credit_score,
        "DTI_Ratio":          dti_ratio,
        "Savings":            savings,
        "Education_Level":    edu_order.index(education),
        "Employment_Status":  employment,
        "Employer_Category":  employer_cat,
        "Loan_Purpose":       loan_purpose,
        "Property_Area":      property_area,
        "Gender":             gender,
        "Marital_Status":     marital_status,
    }])

    # OHE encode
    ohe_cols    = [c for c in CATEGORICAL_COLS if c in input_df.columns]
    ohe_encoded = encoder.transform(input_df[ohe_cols])
    ohe_df      = pd.DataFrame(ohe_encoded, columns=encoder.get_feature_names_out(ohe_cols))
    input_df    = pd.concat([input_df.drop(columns=ohe_cols), ohe_df], axis=1)

    # Feature engineering (same as training)
    input_df["DTI_Ratio_sq"]    = input_df["DTI_Ratio"] ** 2  if "DTI_Ratio" in input_df.columns else 0
    input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2 if "Credit_Score" in input_df.columns else 0
    input_df = input_df.drop(columns=["DTI_Ratio", "Credit_Score"], errors="ignore")

    # Align to scaler's expected features
    expected_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else input_df.columns
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols]

    # Scale & predict
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

    st.divider()

    # Result
    if prediction == 1:
        st.success("## ✅ Loan Approved!")
    else:
        st.error("## ❌ Loan Not Approved")

    # Confidence chart
    if proba is not None:
        col_a, col_b = st.columns([1, 2])
        col_a.metric("Approval Probability", f"{proba[1]*100:.1f}%")
        col_a.metric("Rejection Probability", f"{proba[0]*100:.1f}%")

        fig, ax = plt.subplots(figsize=(5, 2.5))
        bars = ax.barh(["Not Approved", "Approved"], proba,
                       color=["#e74c3c", "#2ecc71"], height=0.5)
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Confidence")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        col_b.pyplot(fig)
        plt.close()
