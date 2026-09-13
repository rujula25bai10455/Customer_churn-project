import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Customer Churn Prediction Engine", layout="wide")

# Check required files
required_files = [
    "src/preprocessor.joblib",
    "models/best_churn_model.joblib",
    "models/optimal_threshold.txt"
]
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    st.error(f"Missing required artifact files: {missing}. Please run steps 3, 4, and 5 first.")
    st.stop()

# 1. Load Artifacts
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("src/preprocessor.joblib")
    model = joblib.load("models/best_churn_model.joblib")
    with open("models/optimal_threshold.txt", "r") as f:
        optimal_threshold = float(f.read().strip())
    return preprocessor, model, optimal_threshold

preprocessor, model, optimal_threshold = load_artifacts()

# App Header
st.title("Enterprise Customer Churn Prediction Engine")
st.markdown("Predict individual customer attrition, risk tier, and diagnostic drivers in real time.")
st.divider()

# Sidebar: User Inputs
st.sidebar.header("Customer Profile & Usage")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city = st.selectbox("City", ["New York", "Los Angeles", "Chicago", "Houston", "Miami"])
    tenure_months = st.number_input("Tenure (Months)", min_value=1, max_value=120, value=12)

with col_sb2:
    plan_type = st.selectbox("Plan Type", ["Basic", "Standard", "Premium"])
    contract_type = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=300.0, value=65.0)

st.sidebar.subheader("Activity & Support")
monthly_usage = st.sidebar.number_input("Monthly Usage (GB / Units)", min_value=0.0, max_value=1000.0, value=150.0)
login_frequency = st.sidebar.number_input("Login Frequency / Month", min_value=1, max_value=100, value=14)
support_tickets = st.sidebar.number_input("Support Tickets Opened", min_value=0, max_value=30, value=2)
avg_session_duration = st.sidebar.number_input("Avg Session Duration (Mins)", min_value=1.0, max_value=180.0, value=25.0)

# 2. Build Single-Row Input & Feature Engineering
input_df = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "city": [city],
    "tenure_months": [tenure_months],
    "plan_type": [plan_type],
    "monthly_charges": [monthly_charges],
    "contract_type": [contract_type],
    "payment_method": [payment_method],
    "monthly_usage": [monthly_usage],
    "login_frequency": [login_frequency],
    "support_tickets": [support_tickets],
    "avg_session_duration": [avg_session_duration]
})

# Derived metrics
input_df["support_ticket_rate"] = input_df["support_tickets"] / np.maximum(input_df["tenure_months"], 1)
input_df["clv_proxy"] = input_df["tenure_months"] * input_df["monthly_charges"]
input_df["usage_to_cost_ratio"] = input_df["monthly_usage"] / np.maximum(input_df["monthly_charges"], 1)
input_df["engagement_intensity"] = input_df["monthly_usage"] / np.maximum(input_df["login_frequency"], 1)
input_df["tenure_cohort"] = pd.cut(
    input_df["tenure_months"],
    bins=[-1, 12, 24, 48, 120],
    labels=["0-12m", "13-24m", "25-48m", "49m+"]
)

# 3. Transform and Predict
processed_input = preprocessor.transform(input_df)
churn_prob = float(model.predict_proba(processed_input)[0, 1])

# 4. Display Results
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction & Risk Tier")
    st.metric(label="Predicted Churn Probability", value=f"{churn_prob:.1%}")

    if churn_prob < 0.30:
        st.success("Risk Status: LOW RISK")
        st.write("**Action Plan:** Customer engagement healthy. Standard retention loyalty flow.")
    elif churn_prob <= 0.70:
        st.warning("Risk Status: MEDIUM RISK")
        st.write("**Action Plan:** Moderate risk. Initiate outreach campaign and usage optimization offers.")
    else:
        st.error("Risk Status: HIGH RISK")
        st.write("**Action Plan:** Urgent retention priority. Direct customer success contact & discount incentives.")

    st.info(f"Cost-Optimized Business Threshold: **{optimal_threshold:.2f}**")

with col2:
    st.subheader("Top Global Drivers from Analysis")
    if os.path.exists("reports/figures/shap_feature_importance.png"):
        st.image("reports/figures/shap_feature_importance.png", caption="Model Feature Drivers (SHAP)")
    else:
        st.write("SHAP summary plot not found in `reports/figures/`.")