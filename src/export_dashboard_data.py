import sqlite3
import pandas as pd
import numpy as np

# 1. Connect to SQLite database
conn = sqlite3.connect("rein_credit_risk.db")

# 2. Join article metadata, qualitative text, and AI covariants
query = """
SELECT 
    a.id AS article_id,
    a.title,
    a.url,
    c.property_sector,
    c.distress_score,
    c.refinancing_risk,
    c.vacancy_pressure,
    c.ai_rationale
FROM article_intelligence a
JOIN credit_risk_covariants c ON a.id = c.article_id
"""

df = pd.read_sql_query(query, conn)
conn.close()

# 3. Attach simulated financial ratios & probability of default
np.random.seed(42)
n_samples = len(df)

df["loan_to_value"] = np.random.normal(loc=0.72, scale=0.08, size=n_samples).clip(0.50, 0.95)
df["debt_service_coverage"] = np.random.normal(loc=1.25, scale=0.20, size=n_samples).clip(0.85, 1.85)

# Calculate calculated Probability of Default (PD %)
raw_pd = (
    0.35 * df["loan_to_value"] 
    - 0.40 * df["debt_service_coverage"] 
    + 0.15 * (df["distress_score"] / 5.0) 
    + 0.10 * df["refinancing_risk"]
)
# Normalize to standard 0% to 100% credit risk rating
df["predicted_default_prob_pct"] = ((raw_pd - raw_pd.min()) / (raw_pd.max() - raw_pd.min()) * 100).round(2)

# Assign Risk Rating Category
def assign_rating(prob):
    if prob >= 70.0: return "High Risk (Watchlist)"
    elif prob >= 40.0: return "Moderate Risk"
    else: return "Low Risk (Performing)"

df["credit_risk_tier"] = df["predicted_default_prob_pct"].apply(assign_rating)

# 4. Export clean dataset to CSV
output_filename = "tableau_cre_credit_risk.csv"
df.to_csv(output_filename, index=False)

print("="*60)
print(f" SUCCESS! Dashboard export ready: '{output_filename}'")
print("="*60)
print(df[["title", "property_sector", "distress_score", "predicted_default_prob_pct", "credit_risk_tier"]].head())