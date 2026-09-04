streamlit run app.py
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

st.title("Telco Customer Churn Prediction")
st.write("Enter customer attributes below to evaluate churn probability.")

tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

input_df = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'PaperlessBilling': [paperless],
    'PaymentMethod': [payment]
})

if st.button("Predict Churn"):
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        
        if pred == 1:
            st.error(f"High Churn Risk! Probability: {prob:.1%}")
        else:
            st.success(f"Low Churn Risk. Probability: {prob:.1%}")
    else:
        st.warning("`model.pkl` not found. Training model now...")
        import train_model
        st.rerun()
