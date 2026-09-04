import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Telco Customer Churn Prediction", page_icon="📊", layout="wide"
)

st.title("📊 Telco Customer Churn Prediction")
st.write("Customer Churn Prediction Using Machine Learning Model ")


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
  file_name = "Telco_Cusomer_Churn.csv"
  df = pd.read_csv(file_name)
  # Convert TotalCharges to numeric
  df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
  # Fill missing values
  df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
  return df


try:
  df = load_data()
except Exception:
  st.error(
      "CSV file not found. Please put 'Telco_Cusomer_Churn.csv' in the same"
      " folder as app.py."
  )
  st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page", ["Dashboard", "Model Comparison", "Prediction"]
)

# ============================================================
# PREPARE DATA
# ============================================================
model_df = df.copy()
# Remove customer ID
if "customerID" in model_df.columns:
  model_df = model_df.drop("customerID", axis=1)

# Convert target
model_df["Churn"] = model_df["Churn"].map({"No": 0, "Yes": 1})

X = model_df.drop("Churn", axis=1)
y = model_df["Churn"]

# ============================================================
# FEATURE TYPES
# ============================================================
numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]
categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ============================================================
# PREPROCESSING
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ============================================================
# MACHINE LEARNING MODELS
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    ),
}


# ============================================================
# TRAIN MODELS
# ============================================================
@st.cache_resource
def train_models(X_train, y_train):
  trained_models = {}
  for name, model in models.items():
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)]
    )
    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline
  return trained_models


trained_models = train_models(X_train, y_train)


# ============================================================
# MODEL EVALUATION
# ============================================================
def evaluate_models():
  results = []
  predictions = {}
  probabilities = {}
  for name, model in trained_models.items():
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    predictions[name] = pred
    probabilities[name] = prob
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
    })
  results_df = pd.DataFrame(results)
  return results_df, predictions, probabilities


results_df, predictions, probabilities = evaluate_models()

# ============================================================
# FIND BEST MODEL
# ============================================================
best_model_name = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
best_accuracy = results_df.loc[results_df["Accuracy"].idxmax(), "Accuracy"]
best_model = trained_models[best_model_name]

# ============================================================
# PAGE 1 - DASHBOARD
# ============================================================
if page == "Dashboard":
  st.header("📈 Telco Customer Churn Dashboard")
  total_customers = len(df)
  churned_customers = (df["Churn"] == "Yes").sum()
  churn_rate = (churned_customers / total_customers) * 100

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Total Customers", f"{total_customers:,}")
  col2.metric("Churned Customers", f"{churned_customers:,}")
  col3.metric("Churn Rate", f"{churn_rate:.2f}%")
  col4.metric("Best Model", best_model_name)

  st.divider()

  # --------------------------------------------------------
  # ENHANCED COLORFUL BAR CHART: Churn Distribution
  # --------------------------------------------------------
  st.subheader("Customer Churn Distribution")
  churn_counts = df["Churn"].value_counts()

  fig, ax = plt.subplots(figsize=(7, 4))
  bar_colors = ["#2ecc71", "#e74c3c"]  # Emerald Green for No, Crimson Red for Yes

  bars = ax.bar(
      churn_counts.index,
      churn_counts.values,
      color=bar_colors,
      width=0.45,
      edgecolor="black",
      linewidth=0.8,
  )

  ax.set_xlabel("Churn Status", fontweight="bold", fontsize=11)
  ax.set_ylabel("Number of Customers", fontweight="bold", fontsize=11)
  ax.set_title(
      "Customer Churn Count Breakdown",
      fontsize=13,
      fontweight="bold",
      pad=15,
  )
  ax.set_ylim(0, max(churn_counts.values) * 1.15)
  ax.grid(axis="y", linestyle="--", alpha=0.5)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  # Value labels above bars
  for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + (total_customers * 0.015),
        f"{yval:,}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

  st.pyplot(fig)

  st.divider()
  st.subheader("Dataset Preview")
  st.dataframe(df.head(10), use_container_width=True)

# ============================================================
# PAGE 2 - MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":
  st.header("🤖 Machine Learning Model Comparison")

  )

  # --------------------------------------------------------
  # RESULTS TABLE
  # --------------------------------------------------------
  st.subheader("Model Performance")
  display_results = results_df.copy()
  metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
  display_results[metric_columns] = (
      display_results[metric_columns] * 100
  ).round(2)
  st.dataframe(display_results, use_container_width=True, hide_index=True)

  # --------------------------------------------------------
  # BEST MODEL BANNER
  # --------------------------------------------------------
  st.success(
      f"🏆 Most Accurate Model: {best_model_name} with"
      f" {best_accuracy * 100:.2f}% accuracy"
  )

  # --------------------------------------------------------
  # ENHANCED COLORFUL BAR CHART: Accuracy Comparison
  # --------------------------------------------------------
  st.subheader("Accuracy Comparison")
  accuracy_data = results_df[["Model", "Accuracy"]].copy()
  accuracy_data["Accuracy"] *= 100

  fig, ax = plt.subplots(figsize=(8, 4.5))
  model_colors = ["#2b5c8f", "#e67e22", "#27ae60"]  # Vibrant distinct palette

  bars = ax.bar(
      accuracy_data["Model"],
      accuracy_data["Accuracy"],
      color=model_colors,
      width=0.45,
      edgecolor="black",
      linewidth=0.8,
  )

  ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=11)
  ax.set_xlabel("Machine Learning Model", fontweight="bold", fontsize=11)
  ax.set_title(
      "Machine Learning Model Accuracy Comparison",
      fontsize=13,
      fontweight="bold",
      pad=15,
  )
  ax.set_ylim(0, 105)
  ax.grid(axis="y", linestyle="--", alpha=0.5)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  # Data value annotations
  for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 2,
        f"{yval:.2f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

  st.pyplot(fig)

  # --------------------------------------------------------
  # ENHANCED COLORFUL BAR CHART: All Performance Metrics
  # --------------------------------------------------------
  st.subheader("Performance Metrics Comparison")

  metrics_plot_df = results_df.set_index("Model") * 100

  fig, ax = plt.subplots(figsize=(10, 5))
  metric_colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#f1c40f"]

  metrics_plot_df.plot(
      kind="bar",
      ax=ax,
      color=metric_colors,
      width=0.7,
      edgecolor="black",
      linewidth=0.5,
  )

  ax.set_ylabel("Score (%)", fontweight="bold", fontsize=11)
  ax.set_xlabel("Machine Learning Model", fontweight="bold", fontsize=11)
  ax.set_title(
      "Comparison across Evaluation Metrics",
      fontsize=13,
      fontweight="bold",
      pad=15,
  )
  ax.set_ylim(0, 110)
  ax.grid(axis="y", linestyle="--", alpha=0.5)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  plt.xticks(rotation=0, fontweight="bold")
  ax.legend(
      title="Metrics",
      bbox_to_anchor=(1.01, 1),
      loc="upper left",
      frameon=True,
  )

  plt.tight_layout()
  st.pyplot(fig)

  # --------------------------------------------------------
  # CONFUSION MATRIX
  # --------------------------------------------------------
  st.subheader("Confusion Matrix")
  selected_model = st.selectbox("Select Model", list(trained_models.keys()))
  cm = confusion_matrix(y_test, predictions[selected_model])

  fig, ax = plt.subplots(figsize=(5, 4))
  cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
  fig.colorbar(cax)

  ax.set_title(
      f"Confusion Matrix - {selected_model}", fontweight="bold", fontsize=12
  )
  ax.set_xlabel("Predicted", fontweight="bold")
  ax.set_ylabel("Actual", fontweight="bold")
  ax.set_xticks([0, 1], ["No Churn", "Churn"])
  ax.set_yticks([0, 1], ["No Churn", "Churn"])

  for i in range(2):
    for j in range(2):
      ax.text(
          j,
          i,
          f"{cm[i, j]:,}",
          ha="center",
          va="center",
          color="white" if cm[i, j] > cm.max() / 2 else "black",
          fontweight="bold",
          fontsize=12,
      )

  plt.tight_layout()
  st.pyplot(fig)

# ============================================================
# PAGE 3 - CUSTOMER PREDICTION
# ============================================================
elif page == "Prediction":
  st.header("🔮 Customer Churn Prediction")
  st.write(
      "Enter customer information below to predict the probability of customer"
      " churn."
  )

  # --------------------------------------------------------
  # SHOW BEST MODEL
  # --------------------------------------------------------
  st.info(
      f"Prediction model: **{best_model_name}** (Accuracy:"
      f" {best_accuracy * 100:.2f}%)"
  )

  # --------------------------------------------------------
  # INPUTS
  # --------------------------------------------------------
  col1, col2 = st.columns(2)
  with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["Yes", "No", "No phone service"]
    )
    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
  with col2:
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

  col3, col4 = st.columns(2)
  with col3:
    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, max_value=200.0, value=70.0
    )
  with col4:
    total_charges = st.number_input(
        "Total Charges", min_value=0.0, max_value=10000.0, value=1000.0
    )

  # --------------------------------------------------------
  # CREATE CUSTOMER DATA
  # --------------------------------------------------------
  customer_data = pd.DataFrame({
      "gender": [gender],
      "SeniorCitizen": [senior_citizen],
      "Partner": [partner],
      "Dependents": [dependents],
      "tenure": [tenure],
      "PhoneService": [phone_service],
      "MultipleLines": [multiple_lines],
      "InternetService": [internet_service],
      "OnlineSecurity": [online_security],
      "OnlineBackup": [online_backup],
      "DeviceProtection": [device_protection],
      "TechSupport": [tech_support],
      "StreamingTV": [streaming_tv],
      "StreamingMovies": [streaming_movies],
      "Contract": [contract],
      "PaperlessBilling": [paperless_billing],
      "PaymentMethod": [payment_method],
      "MonthlyCharges": [monthly_charges],
      "TotalCharges": [total_charges],
  })

  # --------------------------------------------------------
  # PREDICT
  # --------------------------------------------------------
  if st.button("🔮 Predict Customer Churn", use_container_width=True):
    prediction = best_model.predict(customer_data)[0]
    probability = best_model.predict_proba(customer_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
      if prediction == 1:
        st.error("⚠️ HIGH RISK: CUSTOMER MAY CHURN")
      else:
        st.success("✅ LOW RISK: CUSTOMER LIKELY TO STAY")
    with col2:
      st.metric("Churn Probability", f"{probability * 100:.2f}%")

    # Progress bar
    st.write("Customer Churn Risk")
    st.progress(float(probability))

    # Business recommendation
    st.subheader("💡 Recommended Business Action")
    if probability >= 0.70:
      st.warning("""
                **High-risk customer detected.**
                Recommended actions:
                • Contact the customer immediately  
                • Offer a personalised retention discount  
                • Consider a longer-term contract  
                • Provide additional technical support  
                • Review the customer's monthly charges  
                • Offer relevant service upgrades
                """)
    elif probability >= 0.40:
      st.warning("""
                **Medium-risk customer.**
                Recommended actions:
                • Monitor customer behaviour  
                • Provide personalised promotions  
                • Improve customer engagement  
                • Consider loyalty rewards
                """)
    else:
      st.success("""
                **Low-risk customer.**
                Recommended actions:
                • Maintain current service quality  
                • Continue loyalty programmes  
                • Consider suitable upselling opportunities
                """)

    st.subheader("Customer Information")
    st.dataframe(customer_data, use_container_width=True)
