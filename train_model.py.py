import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Connect to SQLite and pull your AI-generated covariants
conn = sqlite3.connect("rein_credit_risk.db")
df_covariants = pd.read_sql_query("SELECT * FROM credit_risk_covariants", conn)
conn.close()

# 2. Simulate baseline commercial loan portfolio metrics matching the 29 assets
np.random.seed(42) # Ensure reproducible results
n_samples = len(df_covariants)

# Generate realistic commercial real estate balance sheet metrics
df_covariants["loan_to_value"] = np.random.normal(loc=0.72, scale=0.08, size=n_samples).clip(0.50, 0.95)
df_covariants["debt_service_coverage"] = np.random.normal(loc=1.25, scale=0.20, size=n_samples).clip(0.85, 1.85)

# Define ground-truth Default Event (1 = Default/Special Servicing, 0 = Performing)
# Driven by high LTV, low DSCR, and elevated AI Distress Score
default_probability = (
    0.35 * df_covariants["loan_to_value"] 
    - 0.40 * df_covariants["debt_service_coverage"] 
    + 0.15 * (df_covariants["distress_score"] / 5.0) 
    + 0.10 * df_covariants["refinancing_risk"]
)
df_covariants["default_event"] = (default_probability > default_probability.median()).astype(int)

# 3. Prepare Feature Matrices for Comparative Modeling
# Baseline Model: Traditional Financials Only
X_baseline = df_covariants[["loan_to_value", "debt_service_coverage"]]

# Augmented Model: Traditional Financials + AI Text Covariants
X_augmented = df_covariants[["loan_to_value", "debt_service_coverage", "distress_score", "refinancing_risk", "vacancy_pressure"]]

y = df_covariants["default_event"]

# Split data into training and testing sets
X_base_train, X_base_test, y_train, y_test = train_test_split(X_baseline, y, test_size=0.3, random_state=42)
X_aug_train, X_aug_test, _, _ = train_test_split(X_augmented, y, test_size=0.3, random_state=42)

# 4. Train Multinomial / Binary Logistic Regression
log_model_base = LogisticRegression()
log_model_base.fit(X_base_train, y_train)
base_preds = log_model_base.predict_proba(X_base_test)[:, 1]

log_model_aug = LogisticRegression()
log_model_aug.fit(X_aug_train, y_train)
aug_preds = log_model_aug.predict_proba(X_aug_test)[:, 1]

# 5. Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_aug_train, y_train)
rf_preds = rf_model.predict_proba(X_aug_test)[:, 1]

# 6. Evaluate and Print Performance Gains
print("="*60)
print(" QUANTITATIVE MODEL PERFORMANCE EVALUATION")
print("="*60)
print(f"Baseline Logistic Regression AUC-ROC : {roc_auc_score(y_test, base_preds):.4f}")
print(f"Augmented (Text-Infused) Logistic AUC : {roc_auc_score(y_test, aug_preds):.4f}")
print(f"Random Forest Classifier AUC-ROC      : {roc_auc_score(y_test, rf_preds):.4f}")
print("="*60)

# Feature Importance from Machine Learning Model
feature_importances = pd.Series(rf_model.feature_importances_, index=X_augmented.columns).sort_values(ascending=False)
print("\n--- ML FEATURE IMPORTANCE RANKING ---")
print(feature_importances)