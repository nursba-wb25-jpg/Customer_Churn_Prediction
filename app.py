import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")


# Train both Random Forest and Logistic Regression models
@st.cache_resource
def train_models():
    df = pd.read_csv("Telco_Cusomer_Churn.csv")

    # Clean numerical columns safely
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
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
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    # Model 1: Random Forest
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    rf_pipeline.fit(X, y)

    # Model 2: Logistic Regression (max_iter=1000 prevents convergence crash)
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr_pipeline.fit(X, y)

    return rf_pipeline, lr_pipeline


st.title("Telco Customer Churn Prediction")
st.write("Enter customer attributes below to evaluate churn probability.")

# Load models with fallback
try:
    rf_model, lr_model = train_models()
except Exception as e:
    st.error(f"Error training models: {e}")
    rf_model, lr_model = None, None

# Inputs
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
    if rf_model is not None and lr_model is not None:
        rf_prob = rf_model.predict_proba(input_df)[0][1]
        lr_prob = lr_model.predict_proba(input_df)[0][1]

        if rf_prob >= 0.5:
            st.error(f"High Churn Risk! Probability: {rf_prob:.1%}")
        else:
            st.success(f"Low Churn Risk. Probability: {rf_prob:.1%}")

        st.markdown("---")

        st.subheader("Machine Learning Model Comparison")
        comparison_df = pd.DataFrame({
            "Model": ["Random Forest", "Logistic Regression"],
            "Predicted Risk": [
                "High Churn" if rf_prob >= 0.5 else "Low Churn",
                "High Churn" if lr_prob >= 0.5 else "Low Churn",
            ],
            "Churn Probability": [f"{rf_prob:.1%}", f"{lr_prob:.1%}"],
        })
        st.table(comparison_df)

        st.subheader("📈 Probability Comparison Chart (%)")
        chart_data = pd.DataFrame(
            [rf_prob * 100, lr_prob * 100],
            index=["Random Forest", "Logistic Regression"],
            columns=["Churn Risk (%)"],
        )
        st.bar_chart(chart_data)
