import sqlite3
import pandas as pd
import json
import time

# Optional: Uncomment and install openai if using OpenAI's API
# from openai import OpenAI
# client = OpenAI(api_key="YOUR_API_KEY_HERE")

# 1. Connect to your SQLite database
conn = sqlite3.connect("rein_credit_risk.db")
cursor = conn.cursor()

# 2. Create the target table for structured numerical covariants
cursor.execute("""
    CREATE TABLE IF NOT EXISTS credit_risk_covariants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER,
        title TEXT,
        distress_score REAL,        -- Scale: -5.0 (High Expansion) to +5.0 (Severe Distress/Default)
        refinancing_risk REAL,      -- Scale: 0.0 (None) to 1.0 (Critical Risk)
        vacancy_pressure REAL,      -- Scale: 0.0 (High Absorption) to 1.0 (High Vacancy/Downsizing)
        property_sector TEXT,       -- Office, Retail, Industrial, Multifamily, Hospitality
        ai_rationale TEXT,
        FOREIGN KEY (article_id) REFERENCES article_intelligence (id)
    )
""")
conn.commit()

# 3. Pull unprocessed raw text articles from your database
df_articles = pd.read_sql_query("SELECT id, title, body_text FROM article_intelligence", conn)

print(f"Loaded {len(df_articles)} raw text records from database for AI feature engineering...\n")

# Define the structured system prompt
SYSTEM_PROMPT = """
You are a Senior Quantitative Credit Risk Analyst at a commercial bank. 
Analyze the provided real estate news article and return ONLY a raw JSON object with no markdown code blocks.

Evaluate the qualitative text and return the following numeric covariants:
1. "distress_score": Float between -5.0 (extreme expansion/growth) and +5.0 (severe financial distress, foreclosure, or default).
2. "refinancing_risk": Float between 0.0 (no risk) and 1.0 (severe debt liquidity or maturity cliff risk).
3. "vacancy_pressure": Float between 0.0 (strong tenant demand) and 1.0 (high vacancy, corporate downsizing, or tenant defaults).
4. "property_sector": String indicating primary sector ("Office", "Retail", "Industrial", "Multifamily", or "Other").
5. "ai_rationale": One concise sentence explaining the economic reasoning.
"""

def generate_mock_scores(title, text):
    """
    Temporary heuristic parser to test your SQL pipeline 
    before attaching your live LLM API key.
    """
    text_lower = text.lower()
    distress = 0.0
    refinancing = 0.2
    vacancy = 0.2
    sector = "Other"

    if "office" in text_lower: sector = "Office"
    elif "apartment" in text_lower or "housing" in text_lower: sector = "Multifamily"
    elif "industrial" in text_lower or "logistics" in text_lower: sector = "Industrial"
    elif "retail" in text_lower or "mall" in text_lower: sector = "Retail"

    if any(k in text_lower for k in ["foreclosed", "distress", "downsize", "knocks", "on pause"]):
        distress += 3.5
        refinancing += 0.5
        vacancy += 0.4
    elif any(k in text_lower for k in ["sells for", "buys", "takes over", "incentives"]):
        distress -= 2.0

    return {
        "distress_score": round(max(min(distress, 5.0), -5.0), 2),
        "refinancing_risk": round(min(refinancing, 1.0), 2),
        "vacancy_pressure": round(min(vacancy, 1.0), 2),
        "property_sector": sector,
        "ai_rationale": f"Heuristic tag based on key commercial real estate indicators in title: {title[:40]}..."
    }

# 4. Loop through articles and populate structured numerical table
processed_count = 0

for idx, row in df_articles.iterrows():
    article_id = row["id"]
    title = row["title"]
    body_text = row["body_text"][:2000] # Pass first 2,000 characters to API

    # --- TO USE LIVE OPENAI API: UNCOMMENT THIS BLOCK ---
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": f"Title: {title}\nText: {body_text}"}
    #     ],
    #     temperature=0.1
    # )
    # scores = json.loads(response.choices[0].message.content)
    # ----------------------------------------------------

    # Using heuristic parser for pipeline testing:
    scores = generate_mock_scores(title, body_text)

    cursor.execute("""
        INSERT INTO credit_risk_covariants 
        (article_id, title, distress_score, refinancing_risk, vacancy_pressure, property_sector, ai_rationale)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        article_id, 
        title, 
        scores["distress_score"], 
        scores["refinancing_risk"], 
        scores["vacancy_pressure"], 
        scores["property_sector"], 
        scores["ai_rationale"]
    ))
    
    conn.commit()
    processed_count += 1
    print(f"[{processed_count}/{len(df_articles)}] Scored ID {article_id} | Sector: {scores['property_sector']} | Distress Score: {scores['distress_score']}")

conn.close()
print("\nSuccess! Unstructured text converted into structured numeric covariants in 'credit_risk_covariants'.")