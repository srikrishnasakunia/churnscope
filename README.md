# ChurnScope — Predicting Telecom Customer Churn with Five Classifiers

**M.Tech (AIML) · Machine Learning · Assignment 2**

- **Live app:** `<paste your Streamlit Community Cloud URL here>`
- **Repository:** https://github.com/srikrishnasakunia/churnscope

---

## a. Problem statement

A telecom operator loses revenue every time a subscriber cancels ("churns"), and keeping an existing customer is far cheaper than winning a new one. The goal is to predict, from a customer's account details and the services they use, **whether that customer will churn (Yes / No)**. That makes it a **binary classification** problem.

Five classical classifiers are trained on the same data and compared on six metrics. The best one is then served through an interactive Streamlit app that scores an uploaded test file **without retraining**.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Telco Customer Churn (IBM sample dataset, published on Kaggle as `blastchar/telco-customer-churn`) |
| Source | Kaggle — https://www.kaggle.com/datasets/blastchar/telco-customer-churn (mirror: IBM GitHub `telco-customer-churn-on-icp4d`) |
| Instances | **7,043** customers (minimum required: 500) |
| Input features | **19** (minimum required: 12). `customerID` is an identifier and is dropped |
| Target | `Churn` — Yes (1,869 · 26.5%) / No (5,174 · 73.5%). The classes are imbalanced |
| Task | Binary classification, positive class = churns |

**Features**

| Group | Columns |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

**Preprocessing** (in `churn_utils.py` plus a scikit-learn `ColumnTransformer` inside each saved pipeline):

1. `TotalCharges` is stored as text and is blank for 11 customers with `tenure = 0`. Those customers have not been billed yet, so the blanks are set to 0 rather than dropping the rows.
2. `SeniorCitizen` (0/1) is treated as a Yes/No category.
3. The 16 categorical columns are one-hot encoded. The 3 numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) are standardised (mean 0, standard deviation 1).
4. Split: **80% train (5,634 rows) / 20% test (1,409 rows)**, stratified on `Churn` with `random_state = 42`. The raw test rows are saved as `test_data.csv`.

Because preprocessing lives inside each saved pipeline, the app accepts the raw CSV exactly as it appears in the original file.

## c. GitHub repository

https://github.com/srikrishnasakunia/churnscope

```
churnscope/
│-- app.py                     Streamlit app (loads saved models, never retrains)
│-- churn_utils.py             shared cleaning + feature lists
│-- requirements.txt
│-- README.md
│-- test_data.csv              held-out 20% test set (1,409 rows, raw format)
│-- data/
│   └-- Telco-Customer-Churn.csv   original dataset
│-- model/
│   │-- train_models.py        trains all 5 models, writes metrics + .joblib files
│   │-- metrics.csv            test-set metrics table
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   └-- svm.joblib
└-- .streamlit/config.toml     app theme
```

**Reproduce:**

```bash
pip install -r requirements.txt
python model/train_models.py     # retrains, rewrites model/*.joblib, metrics.csv, test_data.csv
streamlit run app.py
```

## d. Models used

All five models use the same preprocessing and the same train/test split.

| # | Model | Settings |
|---|---|---|
| 1 | Logistic Regression | L2 penalty, C = 1.0, max_iter = 2000 |
| 2 | Decision Tree Classifier | max_depth = 6, min_samples_leaf = 20 (capped to stop it memorising the training data) |
| 3 | K-Nearest Neighbor Classifier | k = 25, distance-weighted votes |
| 4 | Naive Bayes Classifier | **Gaussian** NB |
| 5 | Support Vector Machine | RBF kernel, C = 1.0, wrapped in sigmoid (Platt) calibration so it outputs probabilities for AUC |

### Comparison table (held-out test set, 1,409 customers)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.8055** | **0.8420** | 0.6572 | 0.5588 | **0.6040** | **0.4790** |
| Decision Tree | 0.7970 | 0.8374 | **0.6679** | 0.4679 | 0.5503 | 0.4357 |
| KNN | 0.7750 | 0.8087 | 0.5817 | 0.5428 | 0.5615 | 0.4109 |
| Naive Bayes | 0.6955 | 0.8075 | 0.4597 | **0.8396** | 0.5941 | 0.4268 |
| Support Vector Machine | 0.7928 | 0.7928 | 0.6614 | 0.4492 | 0.5350 | 0.4205 |

*Bold = best in column. For reference, a "model" that predicts "No churn" for everyone already scores 0.7346 accuracy, so accuracy alone is a weak guide here. MCC and F1 matter more.*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Best on four of six metrics (Accuracy 0.806, AUC 0.842, F1 0.604, MCC 0.479). Train accuracy (0.805) equals test accuracy, so it neither overfits nor underfits. The data suits a linear model: churn risk rises or falls steadily with a few strong signals. Its coefficients show short tenure, month-to-month contracts and fiber-optic internet push towards churn, while two-year contracts and long tenure push away. It catches 209 of 374 churners and wrongly flags 109 loyal customers. |
| **Decision Tree** | Close second on Accuracy (0.797) and AUC (0.837), with the highest Precision (0.668): when it says "will churn", it is right two-thirds of the time. Recall is low (0.468): it misses more than half of the churners. With the depth capped at 6, train and test scores stay close (0.809 vs 0.797). Almost half of its splitting power comes from one feature, *Contract = month-to-month*. It is the easiest model to explain to a business team. |
| **KNN** | Middle of the pack (Accuracy 0.775, MCC 0.411). Distance-weighted voting makes it memorise the training set (train accuracy 0.998 vs test 0.775). After one-hot encoding there are 46 mostly 0/1 dimensions, where "nearest neighbour" distances become less meaningful. It is also the slowest at prediction time, since it compares every new customer with all 5,634 training rows. |
| **Naive Bayes** | The outlier. Lowest Accuracy (0.696) and Precision (0.460), but by far the highest Recall (0.840): it catches 314 of 374 churners. The price is 369 false alarms. It assumes the features are independent of each other, but many overlap here (for example, TotalCharges is roughly tenure × MonthlyCharges, and the "No internet service" values repeat across six columns). That overcounts the evidence and pushes probabilities to extremes. Its F1 (0.594) is second-best, so it suits a cheap retention campaign where missing a churner costs more than an extra discount email. |
| **Support Vector Machine** | Accuracy (0.793) and Precision (0.661) are competitive, but it has the lowest AUC (0.793) and the second-lowest Recall (0.449). It uses a conservative boundary and flags only clear-cut churners. The RBF kernel adds training time without beating the plain linear model, which suggests the classes are already close to linearly separable after encoding. |
| **Overall winner for this dataset** | **Logistic Regression.** It has the highest MCC (0.479), F1, AUC and Accuracy, no overfitting, and the fastest training and prediction, and its coefficients are easy to interpret. If the business goal is to "catch as many churners as possible", the decision threshold can be lowered, or Naive Bayes can serve as a high-recall screener. |

## e. Streamlit app features

| Requirement | Where in the app |
|---|---|
| Dataset upload (CSV) | Sidebar → *Upload test data*. A sample `test_data.csv` can be downloaded from the same sidebar |
| Model selection dropdown | Sidebar → *Choose a model* (all five) |
| Display of evaluation metrics | Six metric tiles for the selected model, plus a side-by-side table of all five models |
| Confusion matrix / classification report | Both are shown for the selected model. The comparison tab adds ROC curves and a bar chart |

The app loads the five `.joblib` pipelines from `model/` and only calls `predict` / `predict_proba`. It never refits on uploaded data. Predictions can be downloaded as a CSV.
