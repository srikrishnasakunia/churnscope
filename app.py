"""ChurnScope — Streamlit front end for the Telco churn classifiers.

The app never retrains. It loads the five saved pipelines from model/,
scores whatever test CSV the user uploads, and reports metrics.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from churn_utils import ALL_FEATURES, MODEL_FILES, TARGET_COL, clean_frame, split_xy

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

st.set_page_config(page_title="ChurnScope · Telco churn models", page_icon="📡", layout="wide")


# ---------- loading & scoring ----------
@st.cache_resource
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


def score(pipe, X, y) -> dict:
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]
    has_both = len(np.unique(y)) == 2
    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": {
            "Accuracy": accuracy_score(y, y_pred),
            "AUC": roc_auc_score(y, y_prob) if has_both else float("nan"),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1": f1_score(y, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y, y_pred),
        },
    }


def plot_confusion(y, y_pred, title):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Stays", "Churns"], yticklabels=["Stays", "Churns"],
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_roc(results, y):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y, r["y_prob"])
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} ({r['metrics']['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on uploaded data", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# ---------- sidebar ----------
st.sidebar.title("📡 ChurnScope")
st.sidebar.caption("Telco Customer Churn · M.Tech AIML · ML Assignment 2")

uploaded = st.sidebar.file_uploader("1. Upload test data (CSV)", type=["csv"])
if SAMPLE_CSV.exists():
    st.sidebar.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")
use_sample = st.sidebar.checkbox("…or use the bundled test_data.csv", value=uploaded is None)

model_name = st.sidebar.selectbox("2. Choose a model", list(MODEL_FILES.keys()))
threshold_note = st.sidebar.caption("Predictions use each model's default 0.5 cut-off.")

# ---------- header ----------
st.title("Will this customer leave?")
st.write(
    "Five classifiers were trained offline on 80% of the IBM Telco Customer Churn data. "
    "Upload the held-out test rows (or any CSV with the same columns plus **Churn**) "
    "and the saved models score them — nothing is retrained here."
)

# ---------- data ----------
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample and SAMPLE_CSV.exists():
    raw = pd.read_csv(SAMPLE_CSV)
    source = "bundled test_data.csv"
else:
    st.info("Upload a CSV from the sidebar to begin.")
    st.stop()

try:
    df = clean_frame(raw)
    X, y = split_xy(df)
except ValueError as err:
    st.error(str(err))
    st.caption(f"Required columns: {', '.join(ALL_FEATURES)} and {TARGET_COL}")
    st.stop()

if y is None or y.isna().any():
    st.error(f"The CSV needs a '{TARGET_COL}' column with Yes/No (or 1/0) values to compute metrics.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Rows scored", f"{len(df):,}")
c2.metric("Actual churners", f"{int(y.sum()):,}")
c3.metric("Churn rate", f"{y.mean():.1%}")
with st.expander(f"Preview of {source}"):
    st.dataframe(raw.head(20), width="stretch")

models = load_models()
results = {name: score(pipe, X, y) for name, pipe in models.items()}

# ---------- tabs ----------
tab_one, tab_all = st.tabs([f"🔍 {model_name}", "📊 Compare all five models"])

with tab_one:
    r = results[model_name]
    m = r["metrics"]
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns([1, 1.25])
    with left:
        st.pyplot(plot_confusion(y, r["y_pred"], f"Confusion matrix — {model_name}"))
        tn, fp, fn, tp = confusion_matrix(y, r["y_pred"], labels=[0, 1]).ravel()
        st.caption(f"Caught {tp} of {tp + fn} churners; {fp} loyal customers were wrongly flagged.")
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y, r["y_pred"], target_names=["Stays (0)", "Churns (1)"],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

    out = raw.copy()
    out["Predicted_Churn"] = np.where(r["y_pred"] == 1, "Yes", "No")
    out["Churn_Probability"] = r["y_prob"].round(4)
    st.download_button("Download predictions as CSV", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].replace('.joblib', '')}.csv",
                       mime="text/csv")

with tab_all:
    table = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T
    table.index.name = "Model"
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8d5"),
                 width="stretch")
    best = table["MCC"].idxmax()
    st.success(f"Best balanced performer on this file (highest MCC): **{best}**")

    a, b = st.columns(2)
    with a:
        st.pyplot(plot_roc(results, y))
    with b:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        table[["Accuracy", "F1", "MCC"]].plot.barh(ax=ax, width=0.75)
        ax.set_xlim(0, 1)
        ax.set_title("Accuracy vs F1 vs MCC", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)

st.caption("Positive class = customer churns. Models: scikit-learn pipelines saved with joblib.")
