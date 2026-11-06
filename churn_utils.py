"""Shared helpers for the Telco churn project.

Used by both the training script (model/train_models.py) and the Streamlit
app (app.py) so that raw CSV rows are cleaned exactly the same way in both.
"""
import pandas as pd

TARGET_COL = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES  # 19 input features

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the raw dataset's quirks and return a cleaned copy.

    * TotalCharges is stored as text and is blank for 11 brand-new customers
      (tenure = 0). A customer who has not been billed yet has paid 0, so we
      fill those blanks with 0 instead of dropping the rows.
    * SeniorCitizen is 0/1 but is really a yes/no category.
    * Churn Yes/No becomes 1/0.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"]).astype(str)
    if TARGET_COL in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET_COL]):
        df[TARGET_COL] =df[TARGET_COL].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0})
    return df


def split_xy(df: pd.DataFrame):
    """Return (X, y) from a cleaned frame. y is None if the target is missing."""
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {missing}")
    X = df[ALL_FEATURES]
    y = df[TARGET_COL].astype(int) if TARGET_COL in df.columns else None
    return X, y
