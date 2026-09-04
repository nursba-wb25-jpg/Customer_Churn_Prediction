import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

# Automatically train and load the model on app startup
@st.cache_resource
def load_or_train_model():
    if not os.path.exists("model.pkl"):
        import train_model
    return joblib.load("model.pkl")

st.title("Telco Customer Churn Prediction")
st.write("Enter customer attributes below to evaluate churn probability.")

# Load model
try:
    model = load_or_train_model()
except Exception as e:
    st.error(f"Error initializing model: {e}")
    model = None

# Input fields
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
    if model is not None:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        
        if pred == 1:
            st.error(f"High Churn Risk! Probability: {prob:.1%}")
        else:
            st.success(f"Low Churn Risk. Probability: {prob:.1%}")
    else:
        st.error("Model is not ready.")
