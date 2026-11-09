# Credit Card Customer Attrition

**Live dashboard:** https://anand-ayush14.github.io/projects/card-attrition/

**Business question:** About 1 in 6 card holders closed their card. Who is about to leave, what are the early-warning signs, and is a retention campaign worth paying for?

**Tools:** SQL (SQLite) · Python · Pandas · Scikit-learn (Gradient Boosting, Random Forest, Logistic Regression, K-Means) · Chart.js

## Data
- *Credit Card Customers* (BankChurners) dataset: 10,127 card holders of a consumer bank with 12 months of activity. Amounts are in USD.
- Inputs: card tier, credit limit, revolving balance, 12-month spend and transaction count, Q4-vs-Q1 change, months inactive, contacts with the bank, products held, and demographics.
- Target: `attrited` (1 = closed the card), 16.1% of customers.
- Cleaning:
  - Renamed columns and dropped leftover model-output and blank columns.
  - Verified unique IDs and no missing values.
  - Kept "Unknown" education, income and marital values as their own category.

## Approach
1. **SQL analysis** (`queries.sql`, 10 queries): attrition by transaction band, activity trend, contacts, products held, revolving balance and months inactive. Also card-tier share of leavers (window function), lost spend by spend quintile (`NTILE` + CTE) and income ranked with `RANK()`.
2. **Features:** average transaction value plus 13 behavioural fields.
   - Gender, education and marital status are excluded.
   - Open-to-buy is dropped because it equals credit limit minus revolving balance.
3. **Model comparison:** Logistic Regression, Random Forest and Gradient Boosting, using 5-fold stratified CV on a 75% training split. The champion is chosen on CV PR-AUC and confirmed on the 25% test set.
4. **Explainability:** permutation importance on the champion.
5. **Segmentation:** K-Means on 8 scaled behavioural features, with k chosen by silhouette score.
6. **Campaign economics:** a profit curve for targeting the top-scored customers, under stated assumptions:
   - $50 per offer.
   - 30% of real leavers are saved.
   - A saved customer is worth 2 years of 2% interchange on spend plus 18% APR on revolving balance.

## Key results
| Metric | Value |
|---|---|
| Attrition rate | 16.1% |
| Q4 transactions fell >50% vs stable | 50.9% vs 7.3% attrition |
| $0 revolving balance vs $1,000–1,999 | 36.2% vs 4.6% attrition |
| 1–2 products vs 6 products | ~26–28% vs 10.5% attrition |
| Customers with 6 bank contacts | 100% left (54 customers) |
| Test PR-AUC / ROC-AUC: Gradient Boosting | 0.967 / 0.993 |
| Test PR-AUC: Random Forest / Logistic Regression | 0.949 / 0.748 |
| Leavers in the top 20% of scores | 97.1% |
| Best campaign (test set) | target the top 17.3%: +$20,089, vs −$80,944 for a blanket offer |

## Recommendations
1. Run a monthly alert for customers whose quarterly transaction count drops by more than 30%.
2. Target only the top ~17% by model score; blanket offers lose money.
3. Re-engage "Fading users" (low use, almost no balance) with spend-based cashback.
4. Cross-sell a second and third product; attrition roughly halves.
5. Route customers with 4+ service contacts to a senior retention team.

## Run it
```bash
pip install pandas numpy scikit-learn
python analysis.py        # downloads the data if BankChurners.csv is not present
```
`analysis.py` prints every SQL result and model metric, and writes `results.json`, which feeds the dashboard (`index.html`).
