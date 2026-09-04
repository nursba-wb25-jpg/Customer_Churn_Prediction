# ============================================
# BMDS2003 DATA SCIENCE
# TELCO CUSTOMER CHURN PREDICTION
# ============================================

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings("ignore")

def main():
    # 1. LOAD AND INSPECT DATA
    data_path = "Telco_Cusomer_Churn.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{data_path}'. Please ensure the CSV file is in the working directory.")
        return

    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nSummary Statistics:")
    print(df.describe(include="all").T)

    # Missing values
    missing = df.isnull().sum()
    missing_df = pd.DataFrame({
        "Missing Values": missing,
        "Percentage": (missing / len(df)) * 100
    })
    print("\nMissing Values Summary:")
    print(missing_df[missing_df["Missing Values"] > 0])
    print("Duplicate rows:", df.duplicated().sum())

    # Check target variable
    print("\nTarget Class Counts:")
    print(df["Churn"].value_counts())
    print("\nTarget Distribution (%):")
    print(df["Churn"].value_counts(normalize=True) * 100)

    # 2. EXPLORATORY DATA ANALYSIS (EDA)
    # Churn Distribution
    plt.figure(figsize=(6, 4))
    df["Churn"].value_counts().plot(kind="bar")
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("churn_distribution.png")
    plt.close()

    # Tenure vs Churn
    plt.figure(figsize=(8, 5))
    df.boxplot(column="tenure", by="Churn")
    plt.title("Tenure Distribution by Churn Status")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Tenure")
    plt.tight_layout()
    plt.savefig("tenure_vs_churn.png")
    plt.close()

    # Monthly Charges vs Churn
    plt.figure(figsize=(8, 5))
    df.boxplot(column="MonthlyCharges", by="Churn")
    plt.title("Monthly Charges by Churn Status")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig("monthly_charges_vs_churn.png")
    plt.close()

    # Contract vs Churn
    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    contract_churn.plot(kind="bar", figsize=(8, 5))
    plt.title("Churn Percentage by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.savefig("contract_vs_churn.png")
    plt.close()

    # 3. DATA PREPARATION
    df = df.drop(columns=["customerID"], errors="ignore")

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Target Encoding
    if df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["Churn"])

    # Separate features X and target y
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Categorical and Numerical variables
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # 4. PREPROCESSING PIPELINE
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # 5. MODEL TRAINING & EVALUATION
    def evaluate_model(name, y_true, y_pred, y_prob):
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_prob)
        }

    # Model 1: Logistic Regression
    logistic_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    logistic_model.fit(X_train, y_train)
    y_pred_lr = logistic_model.predict(X_test)
    y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]

    # Model 2: Decision Tree
    decision_tree = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42, max_depth=5))
    ])
    decision_tree.fit(X_train, y_train)
    y_pred_dt = decision_tree.predict(X_test)
    y_prob_dt = decision_tree.predict_proba(X_test)[:, 1]

    # Model 3: Random Forest
    random_forest = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
    ])
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    y_prob_rf = random_forest.predict_proba(X_test)[:, 1]

    results = [
        evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr),
        evaluate_model("Decision Tree", y_test, y_pred_dt, y_prob_dt),
        evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)
    ]

    results_df = pd.DataFrame(results)
    print("\n--- Model Evaluation Baseline ---")
    print(results_df.sort_values(by="F1-Score", ascending=False).to_string(index=False))

    # 6. HYPERPARAMETER TUNING
    print("\nTuning Hyperparameters for Random Forest...")
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("\nBest Hyperparameters:")
    print(grid_search.best_params_)
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

    best_rf = grid_search.best_estimator_
    y_pred_best = best_rf.predict(X_test)
    y_prob_best = best_rf.predict_proba(X_test)[:, 1]

    print("\n--- Final Model Classification Report ---")
    print(classification_report(y_test, y_pred_best))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_best):.4f}")

    # Save model
    joblib.dump(best_rf, "telco_churn_model.pkl")
    print("\nModel pipeline saved to 'telco_churn_model.pkl'")

if __name__ == "__main__":
    main()
