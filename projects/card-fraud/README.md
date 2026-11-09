# Card Transaction Fraud Detection

**Live dashboard:** https://anand-ayush14.github.io/projects/card-fraud/

**Business question:** Only about 1 in 600 card transactions is fraud, and analysts can't review everything. Which transactions should they review first, and how much fraud does a fixed daily review budget stop?

**Tools:** SQL (SQLite) · Python · Pandas · Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, Isolation Forest) · Chart.js

## Data
- *Credit Card Fraud Detection* dataset (Worldline & ULB Machine Learning Group; Dal Pozzolo et al., 2015): 284,807 European card transactions over 2 days in September 2013, 492 of them fraud.
- `V1`–`V28` are PCA components: the bank anonymised the raw fields for confidentiality. Other fields: `Time` (seconds since the first transaction), `Amount` (EUR) and `Class` (1 = fraud).
- Cleaning: removed 1,081 exact duplicate rows, leaving 283,726 transactions and 473 frauds. Verified no missing values and sorted by time.
- The 100 MB CSV is not stored in this repo. `analysis.py` downloads it.

## Approach
1. **SQL analysis** (`queries.sql`, 9 queries): fraud rate by hour, amount band and night vs day, plus the amount profile.
   - Hourly timeline with a running total and 3-hour moving average (window functions).
   - Largest frauds with `RANK()`.
   - Fraud bursts using `LAG()` to measure the gap to the previous fraud.
2. **Time-based split:** train on the first 75% of transactions and test on the last 9.3 hours. A fraud model always scores future transactions, so a random split would be optimistic.
3. **Model comparison:**
   - Logistic Regression, Random Forest and Gradient Boosting, compared with 3-fold time-series CV on **PR-AUC** (ROC-AUC flatters models when events are this rare).
   - **Isolation Forest** as an unsupervised benchmark that uses no fraud labels.
4. **Operating policy:**
   - Alert-budget table: review the top 0.05%–1% of transactions.
   - Cost curve: €5 per alert reviewed (assumption) plus the value of fraud missed.
   - Permutation importance to show which signals drive alerts.

## Key results
| Metric | Value |
|---|---|
| Fraud rate | 0.167% of transactions |
| Night (low-volume hours) vs day fraud rate | 0.48% vs 0.14% |
| €0 "card test" transactions | 1.4% fraud (about 8× the base rate) |
| Frauds within 10 minutes of another fraud | 84% |
| Test PR-AUC: Random Forest / Gradient Boosting / Logistic Regression | 0.822 / 0.766 / 0.753 |
| Isolation Forest (no labels) | PR-AUC 0.044, despite ROC-AUC 0.94 |
| Review the top 0.1% (~181 alerts a day) | catches 74% of fraud at 99% precision |
| Cost-optimal policy (test window) | ~170 alerts stop €9,452 of €10,838 fraud; net saving €8,602 |

## Recommendations
1. Review the top 0.1–0.25% of transactions by score in real time.
2. Add velocity rules: fraud clusters within minutes.
3. Treat €0 authorisations as card tests and step up authentication on the next purchase.
4. Tighten night-time monitoring.
5. Retrain weekly on confirmed fraud and keep Isolation Forest as a check for new patterns.

## Run it
```bash
pip install pandas numpy scikit-learn
python analysis.py        # about 4 minutes; downloads the data if creditcard.csv is not present
```
`analysis.py` prints every SQL result and model metric, and writes `results.json`, which feeds the dashboard (`index.html`).
