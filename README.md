Title: Customer Churn Prediction
A complete end-to-end machine learning project to predict customer churn using the Telco Customer Churn dataset. The pipeline covers preprocessing, EDA, feature engineering, model training, tuning, evaluation, explainability and deployment.
1. Methodology
Data Collection	→	Preprocessing & Feature Eng.	→	SMOTE + Model Training	→	Hyperparameter Tuning	→	Evaluation & Explainability	→	Deployment

2. Description
●	Dataset = Telco Customer Churn (7,043 customers, 21 features)
●	Target column = Churn (Yes / No)
●	Best Model = XGBoost (tuned via GridSearchCV, 5-fold CV)
●	Best AUC-ROC = 0.91
●	Class imbalance handled with SMOTE oversampling
●	Other information: SHAP used for feature-level explainability; model + scaler saved with joblib
3. Model Performance
Model	AUC-ROC	Result
Logistic Regression (baseline)	0.84	–
Random Forest	0.87	–
XGBoost (tuned)	0.91	✓ Best

4. Top Churn Factors (SHAP)
●	Tenure — short-tenure customers churn more
●	Monthly charges — higher charges correlate with churn
●	Contract type — month-to-month contracts have the highest churn
●	Internet service — fiber optic customers show more churn
●	Tech support — customers without tech support are at higher risk
5. Getting Started
●	Clone the repository:
◦	git clone https://github.com/your-username/customer-churn-prediction.git
●	Install dependencies:
◦	pip install -r requirements.txt
●	Run the notebook:
◦	Open notebooks/churn_prediction.ipynb in Colab or Jupyter and run all cells in order.
6. Tech Stack
●	pandas, numpy, scikit-learn
●	xgboost, imbalanced-learn (SMOTE)
●	shap — model explainability
●	matplotlib, seaborn — visualisation
●	joblib — model persistence
7. Author
Deeksha Sharma — end-to-end data science workflow from raw data ingestion through preprocessing, modeling, evaluation, and deployment.


