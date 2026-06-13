# Customer Churn Prediction

A complete end-to-end machine learning project to predict customer churn using the **Telco Customer Churn** dataset. The pipeline covers data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter tuning, evaluation, and model deployment.

---

##  Problem Statement

Customer churn — when a customer stops using a service — is a major business problem. Identifying at-risk customers early allows companies to take proactive retention steps. This project builds a binary classification model to predict whether a customer will churn (`Yes`) or stay (`No`).

---

##  Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── customer.csv   # Raw dataset
│
├── notebooks/
│   └── churn_prediction.ipynb                  # Main Colab notebook
│
├── models/
│   ├── churn_model.pkl                         # Saved XGBoost model
│   └── scaler.pkl                              # Saved StandardScaler
│
├── requirements.txt                            # Python dependencies
└── README.md
```

---

## Dataset

- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Target column:** `Churn` (Yes / No)
- **Features include:** tenure, monthly charges, contract type, internet service, payment method, and more.

### Load directly (no download needed)

```python
import pandas as pd

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
```

---

## Pipeline Overview

```
Data Collection → EDA → Preprocessing → Feature Engineering
      → SMOTE (class balancing) → Model Training → Evaluation → Deployment
```

### Steps

| Step | Description |
|------|-------------|
| **EDA** | Visualise churn distribution, correlations, missing values |
| **Preprocessing** | Encode categoricals, fix data types, scale features |
| **SMOTE** | Oversample minority class to handle class imbalance |
| **Model Training** | Logistic Regression, Random Forest, XGBoost |
| **Tuning** | GridSearchCV with 5-fold cross-validation |
| **Evaluation** | AUC-ROC, F1-Score, Confusion Matrix, ROC Curve |
| **Explainability** | SHAP values for feature importance |
| **Deployment** | Save model with `joblib`, load and predict on new data |

---

##  Models Trained

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble of decision trees |
| **XGBoost** ✅ | Best performer — gradient boosted trees |

---

## Results

| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | ~0.84 |
| Random Forest | ~0.87 |
| **XGBoost (tuned)** | **~0.91** |

> Results may vary slightly depending on random seed and SMOTE sampling.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `notebooks/churn_prediction.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter and run all cells in order.

---

##  Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
matplotlib
seaborn
joblib
```

Install all at once:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn joblib
```

---

##  Key Code Snippets

### Handle class imbalance with SMOTE

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```

### Train XGBoost

```python
from xgboost import XGBClassifier

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_res, y_train_res)
```

### Evaluate with AUC-ROC

```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"AUC-ROC: {auc:.4f}")
```

### Explain predictions with SHAP

```python
import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### Save and load model

```python
import joblib

# Save
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Load and predict
model  = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

new_customer = scaler.transform([sample_data])
churn_prob   = model.predict_proba(new_customer)[0][1]
print(f"Churn probability: {churn_prob:.2%}")
```

---

##  Key Metrics Explained

| Metric | Why it matters |
|--------|----------------|
| **AUC-ROC** | Measures how well the model separates churners from non-churners |
| **Recall (churn class)** | Missing a churner is costly — high recall minimises false negatives |
| **F1-Score** | Balances precision and recall under class imbalance |
| **SHAP values** | Explains *why* each customer is flagged for churn |

---

##  Top Churn Factors (from SHAP)

Based on feature importance analysis:

- **Tenure** — Short-tenure customers churn more
- **Monthly charges** — Higher charges correlate with churn
- **Contract type** — Month-to-month contracts have highest churn
- **Internet service** — Fiber optic customers show more churn
- **Tech support** — Customers without tech support are at higher risk

---

##  Future Improvements

- [ ] Add a Streamlit or Flask web app for live predictions
- [ ] Experiment with LightGBM and CatBoost
- [ ] Add time-series churn analysis (survival analysis)
- [ ] Deploy model as a REST API using FastAPI
- [ ] Automate retraining pipeline with new data

---

## 👩‍💻 Author

Deeksha Sharma
The project covers the complete data science workflow — from raw data ingestion and preprocessing to model training, evaluation, and deployment.

---


