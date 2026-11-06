"""Train the five classifiers on the Telco Customer Churn dataset.

Run from the project root:
    python model/train_models.py

Outputs
    model/*.joblib         one fitted pipeline (preprocessing + classifier) per model
    model/metrics.csv      evaluation metrics on the held-out test set
    test_data.csv          the raw held-out 20% test rows (upload these in the app)
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from churn_utils import (CATEGORICAL_FEATURES, MODEL_FILES, NUMERIC_FEATURES,  # noqa: E402
                         clean_frame, split_xy)

RAW_CSV = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"
SEED = 42


def make_preprocessor() -> ColumnTransformer:
    # One-hot the 16 categorical columns, standardise the 3 numeric ones.
    # Dense output so GaussianNB can use it too.
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])


def make_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
        # Depth capped: an unrestricted tree memorises the training set.
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "Naive Bayes (Gaussian)": GaussianNB(),
        # RBF-kernel SVM. SVC has no native probabilities, so it is wrapped in
        # Platt (sigmoid) calibration to get the scores needed for AUC.
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, random_state=SEED), method="sigmoid", cv=5, ensemble=False),
    }


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    raw = pd.read_csv(RAW_CSV)
    print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns")

    # Split the RAW rows first so test_data.csv looks exactly like the source file.
    raw_train, raw_test = train_test_split(raw, test_size=0.2, stratify=raw["Churn"], random_state=SEED)
    raw_test.to_csv(ROOT / "test_data.csv", index=False)

    X_train, y_train = split_xy(clean_frame(raw_train))
    X_test, y_test = split_xy(clean_frame(raw_test))
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | churn rate (test): {y_test.mean():.3f}")

    rows = []
    for name, clf in make_models().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        scores = evaluate(y_test, y_pred, y_prob)
        rows.append({"Model": name, **scores})
        joblib.dump(pipe, MODEL_DIR / MODEL_FILES[name])
        print(f"{name:<24} " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print("\n" + metrics.to_markdown(index=False))


if __name__ == "__main__":
    main()
