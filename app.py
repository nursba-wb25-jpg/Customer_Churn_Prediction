import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telco Customer Churn Prediction")
st.write(
    "Machine Learning Model Comparison and Customer Churn Prediction"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    file_name = "Telco_Cusomer_Churn.csv"

    df = pd.read_csv(file_name)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    return df


try:
    df = load_data()

except Exception:
    st.error(
        "CSV file not found. Please put "
        "'Telco_Cusomer_Churn.csv' in the same folder as app.py."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Model Comparison",
        "Prediction"
    ]
)


# ============================================================
# PREPARE DATA
# ============================================================

model_df = df.copy()

# Remove customer ID
if "customerID" in model_df.columns:
    model_df = model_df.drop(
        "customerID",
        axis=1
    )

# Convert target
model_df["Churn"] = model_df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = model_df.drop(
    "Churn",
    axis=1
)

y = model_df["Churn"]


# ============================================================
# FEATURE TYPES
# ============================================================

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
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
    "PaymentMethod"
]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            StandardScaler(),
            numerical_features
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(X_train, y_train):

    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        trained_models[name] = pipeline

    return trained_models


trained_models = train_models(
    X_train,
    y_train
)


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_models():

    results = []

    predictions = {}

    probabilities = {}

    for name, model in trained_models.items():

        pred = model.predict(X_test)

        prob = model.predict_proba(
            X_test
        )[:, 1]

        predictions[name] = pred

        probabilities[name] = prob

        results.append({

            "Model": name,

            "Accuracy":
                accuracy_score(
                    y_test,
                    pred
                ),

            "Precision":
                precision_score(
                    y_test,
                    pred,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_test,
                    pred,
                    zero_division=0
                ),

            "F1 Score":
                f1_score(
                    y_test,
                    pred,
                    zero_division=0
                ),

            "ROC-AUC":
                roc_auc_score(
                    y_test,
                    prob
                )
        })

    results_df = pd.DataFrame(
        results
    )

    return (
        results_df,
        predictions,
        probabilities
    )


results_df, predictions, probabilities = evaluate_models()


# ============================================================
# FIND BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model"
]

best_accuracy = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Accuracy"
]

best_model = trained_models[
    best_model_name
]


# ============================================================
# PAGE 1 - DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📈 Telco Customer Churn Dashboard")

    total_customers = len(df)

    churned_customers = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churned_customers /
        total_customers
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    col4.metric(
        "Best Model",
        best_model_name
    )

    st.divider()

    # Churn distribution

    st.subheader(
        "Customer Churn Distribution"
    )

    churn_counts = df[
        "Churn"
    ].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        churn_counts.index,
        churn_counts.values
    )

    ax.set_xlabel(
        "Churn Status"
    )

    ax.set_ylabel(
        "Number of Customers"
    )

    ax.set_title(
        "Customer Churn Distribution"
    )

    st.pyplot(fig)

    st.divider()

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# PAGE 2 - MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header(
        "🤖 Machine Learning Model Comparison"
    )

    st.write(
        "Three classification models are trained "
        "and evaluated using the same testing dataset."
    )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    st.subheader(
        "Model Performance"
    )

    display_results = results_df.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    display_results[
        metric_columns
    ] = (
        display_results[
            metric_columns
        ] * 100
    ).round(2)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    st.success(
        f"🏆 Most Accurate Model: "
        f"{best_model_name} "
        f"with {best_accuracy * 100:.2f}% accuracy"
    )

    # --------------------------------------------------------
    # ACCURACY CHART
    # --------------------------------------------------------

    st.subheader(
        "Accuracy Comparison"
    )

    accuracy_data = results_df[
        ["Model", "Accuracy"]
    ].copy()

    accuracy_data["Accuracy"] *= 100

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        accuracy_data["Model"],
        accuracy_data["Accuracy"]
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_xlabel(
        "Machine Learning Model"
    )

    ax.set_title(
        "Machine Learning Model Accuracy Comparison"
    )

    ax.set_ylim(
        0,
        100
    )

    for i, value in enumerate(
        accuracy_data["Accuracy"]
    ):

        ax.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha="center"
        )

    st.pyplot(fig)

    # --------------------------------------------------------
    # ALL METRICS
    # --------------------------------------------------------

    st.subheader(
        "Performance Metrics Comparison"
    )

    metrics_long = results_df.copy()

    metrics_long = metrics_long.set_index(
        "Model"
    )

    metrics_long *= 100

    st.bar_chart(
        metrics_long
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )

    selected_model = st.selectbox(
        "Select Model",
        list(trained_models.keys())
    )

    cm = confusion_matrix(
        y_test,
        predictions[selected_model]
    )

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_title(
        f"Confusion Matrix - {selected_model}"
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_xticks(
        [0, 1],
        ["No Churn", "Churn"]
    )

    ax.set_yticks(
        [0, 1],
        ["No Churn", "Churn"]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)


# ============================================================
# PAGE 3 - CUSTOMER PREDICTION
# ============================================================

elif page == "Prediction":

    st.header(
        "Customer Churn Prediction"
    )

    st.write(
        "Enter customer information below to predict "
        "the probability of customer churn."
    )

    # --------------------------------------------------------
    # SHOW BEST MODEL
    # --------------------------------------------------------

    st.info(
        f"Prediction model: **{best_model_name}** "
        f"(Accuracy: {best_accuracy * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (months)",
            0,
            72,
            12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col2:

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col3, col4 = st.columns(2)

    with col3:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

    with col4:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0
        )

    # --------------------------------------------------------
    # CREATE CUSTOMER DATA
    # --------------------------------------------------------

    customer_data = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [
            senior_citizen
        ],

        "Partner": [partner],

        "Dependents": [
            dependents
        ],

        "tenure": [tenure],

        "PhoneService": [
            phone_service
        ],

        "MultipleLines": [
            multiple_lines
        ],

        "InternetService": [
            internet_service
        ],

        "OnlineSecurity": [
            online_security
        ],

        "OnlineBackup": [
            online_backup
        ],

        "DeviceProtection": [
            device_protection
        ],

        "TechSupport": [
            tech_support
        ],

        "StreamingTV": [
            streaming_tv
        ],

        "StreamingMovies": [
            streaming_movies
        ],

        "Contract": [
            contract
        ],

        "PaperlessBilling": [
            paperless_billing
        ],

        "PaymentMethod": [
            payment_method
        ],

        "MonthlyCharges": [
            monthly_charges
        ],

        "TotalCharges": [
            total_charges
        ]
    })

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    ):

        prediction = best_model.predict(
            customer_data
        )[0]

        probability = best_model.predict_proba(
            customer_data
        )[0][1]

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.error(
                    "HIGH RISK: CUSTOMER MAY CHURN"
                )

            else:

                st.success(
                    "LOW RISK: CUSTOMER LIKELY TO STAY"
                )

        with col2:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

        # Progress bar

        st.write(
            "Customer Churn Risk"
        )

        st.progress(
            float(probability)
        )

        # Business recommendation

        st.subheader(
            "💡 Recommended Business Action"
        )

        if probability >= 0.70:

            st.warning(
                """
                **High-risk customer detected.**

                Recommended actions:

                • Contact the customer immediately  
                • Offer a personalised retention discount  
                • Consider a longer-term contract  
                • Provide additional technical support  
                • Review the customer's monthly charges  
                • Offer relevant service upgrades
                """
            )

        elif probability >= 0.40:

            st.warning(
                """
                **Medium-risk customer.**

                Recommended actions:

                • Monitor customer behaviour  
                • Provide personalised promotions  
                • Improve customer engagement  
                • Consider loyalty rewards
                """
            )

        else:

            st.success(
                """
                **Low-risk customer.**

                Recommended actions:

                • Maintain current service quality  
                • Continue loyalty programmes  
                • Consider suitable upselling opportunities
                """
            )

        st.subheader(
            "Customer Information"
        )

        st.dataframe(
            customer_data,
            use_container_width=True
        )
