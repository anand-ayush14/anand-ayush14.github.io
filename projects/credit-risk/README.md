# Credit Card Default Risk Analysis

**Live dashboard:** https://anand-ayush14.github.io/projects/credit-risk/

**Business question:** Which credit card customers are most likely to miss next month's payment, and how should a card issuer act on that risk before it becomes a loss?

**Tools:** SQL (SQLite) · Python · Pandas · Scikit-learn · Chart.js

## Data
- UCI *Default of Credit Card Clients* dataset (Yeh & Lien, 2009): 30,000 card holders in Taiwan, April–September 2005, amounts in NT$.
- 24 inputs: credit limit, age, 6 months of repayment status, bill amounts and payments. Target: `defaulted` (1 = missed the next payment).
- Cleaning: undocumented education codes (0, 5, 6) merged into "other" and marriage code 0 into "other". Verified no missing values and unique IDs.

## Approach
1. **SQL analysis** (`queries.sql`, 8 queries): portfolio KPIs, default rate by repayment status, limit band, utilisation, age band (with a window function for share of portfolio), education, zero-payment customers, and top exposures ranked with `RANK()`.
2. **Feature engineering:** utilisation (bill ÷ limit), worst months late, number of late months, share of bills repaid.
3. **Model comparison:** Logistic Regression, Random Forest and Gradient Boosting, using 5-fold stratified cross-validation on a 75% training split.
4. **Champion/explainer:** Gradient Boosting (best AUC) ranks customers into risk tiers. Logistic Regression coefficients explain which factors drive risk.
5. **Fairness:** sex, marital status and education were excluded from the model. Accuracy was unchanged.

## Key results
| Metric | Value |
|---|---|
| Default rate (base) | 22.1% |
| On-time payers vs 2 months late | 13.8% vs 69.1% default |
| Limit < NT$50K vs NT$400K+ | 36.1% vs 12.0% default |
| Paid nothing on a positive bill | 39.8% default |
| Model ROC-AUC (test): Gradient Boosting / Random Forest / Logistic Regression | 0.766 / 0.767 / 0.732 |
| Top 10% risk tier default rate | 69.9% (vs 11.3% in the bottom 60%) |
| Share of defaulters in the top 20% of scores | 49.7% |

## Recommendations
1. Early-warning outreach after the **first** missed payment (default risk is ~2.5× the on-time rate).
2. Weekly review of the top-10% risk tier: reduce limits or pause new credit.
3. Raise limits for low-limit accounts only after six months of clean repayment.
4. Flag zero-payment customers for collections contact within days.
5. Monitor model drift and fairness monthly, and keep demographic attributes out of decisions.

## Run it
```bash
pip install pandas scikit-learn
python analysis.py        # prints all SQL results and model metrics, writes results.json
```

## Files
- `queries.sql`: all SQL queries
- `analysis.py`: cleaning, SQL execution, features, model comparison, risk tiers
- `results.json`: outputs used by the dashboard
- `index.html`: interactive dashboard

*Author: Ayush Anand. Portfolio: https://anand-ayush14.github.io*
