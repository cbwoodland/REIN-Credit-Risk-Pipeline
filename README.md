\# Quantifying Alternative Data: Replicating the REIN Framework for Commercial Credit Risk



!\[Python](https://img.shields.io/badge/Python-3.14+-blue.svg)

!\[SQLite](https://img.shields.io/badge/SQLite-3.0+-green.svg)

!\[scikit-learn](https://img.shields.io/badge/scikit--learn-1.9+-orange.svg)

!\[License](https://img.shields.io/badge/License-MIT-purple.svg)



\## Executive Summary



Standard quantitative risk models rely heavily on backward-looking financial statements and lagging macroeconomic indicators. Inspired by the \*\*Regional Economic Information Network (REIN)\*\* framework developed by the Federal Reserve Bank of Atlanta—which systematically gathers qualitative, on-the-ground business intelligence to inform monetary policy—this project builds an end-to-end data pipeline that converts qualitative commercial real estate (CRE) narrative data into predictive financial risk signals.



By scraping unstructured market commentary, extracting numerical sentiment features via a Generative AI feature engineering pipeline, and training an augmented \*\*Logistic Regression\*\* and \*\*Random Forest Classifier\*\*, this model demonstrates that incorporating real-time qualitative risk indicators provides a statistically significant leading edge in predicting Probability of Default ($PD$) across commercial debt portfolios.



\### 📊 Interactive Dashboard Preview

\* \*\*Live Tableau Cockpit:\*\* \[View Interactive Dashboard](https://public.tableau.com/views/EnterpriseCreditRisk/Sheet1)



!\[Dashboard Preview](dashboard\_preview.png)



\---



\## Technical \& Modeling Architecture

