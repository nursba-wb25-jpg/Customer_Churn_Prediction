import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")


# Train and cache the model directly in memory when the app loads
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("Telco_Cusomer_Churn.csv")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn"] = df["Churn"].apply(
        lambda x: 1 if str(x).strip().lower() in ["yes", "1"] else 0
    )

    features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    X = df[features]
    y = df["Churn"]

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = ["Contract", "PaperlessBilling", "PaymentMethod"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    pipeline.fit(X, y)
    return pipeline


st.title("Telco Customer Churn Prediction")
st.write("Enter customer attributes below to evaluate churn probability.")

# Load model
try:
    model = load_and_train_model()
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    model = None

# Form Inputs
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input(
    "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0
)
total_charges = st.number_input(
    "Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0
)

contract = st.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

input_df = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract],
    "PaperlessBilling": [paperless],
    "PaymentMethod": [payment],
})

if st.button("Predict Churn"):
    if model is not None:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"High Churn Risk! Probability: {prob:.1%}")
        else:
            st.success(f"Low Churn Risk. Probability: {prob:.1%}")
