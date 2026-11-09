# Corporate Card Expense Copilot

**Live dashboard:** https://anand-ayush14.github.io/projects/expense-copilot/

American Express is building an AI expense-management platform for business card customers (it acquired Center in 2025 and agreed to buy Hyper in 2026). This prototype does the core jobs on **real** company-card data:

1. **Categorise** every purchase (from the merchant category, and from the vendor name alone with ML)
2. **Check policy**: over limit, just under a limit, split purchases, duplicate charges, sensitive merchants, weekend, round amounts
3. **Rank** a short review queue for managers, with a plain-English reason for each flag

## Key findings (Jun 2023 – Sep 2026, 125,615 transactions, $110.9M spend, 79 agencies)

| Finding | Number |
|---|---|
| Purchases at $4,750–5,000 vs $5,000–5,250 | **29.7×** more just under the limit |
| Possible split purchases (same agency + vendor + day, total > $5,000) | **2,050 groups**, $16.9M |
| Possible duplicate charges (same vendor, day, exact amount) | 9,439 extra charges, $5.9M |
| Review queue (top 1% by risk score) | **1,203 purchases**, $4.7M |
| Vendor-name model accuracy on unseen vendors | **68.3%** (majority baseline 23.0%); 92.5% when ≥60% confident |
| Benford first-digit test, all purchases | MAD 0.0075 (acceptable); 5+ agencies show nonconformity |

**Context matters:** the agency with the highest flag rate (Office of the Tenant Advocate) spends mostly at DC hotels, which looks like temporary housing for displaced tenants. Flags are leads for a reviewer, not findings of wrongdoing, and the copilot should learn approved exceptions.

## Data

- [DC Open Data: Purchase Card Transactions](https://opendata.dc.gov/datasets/DCGIS::purchase-card-transactions), licence CC BY 4.0, pulled from the city's ArcGIS REST service (`Public_Service_WebMercator/MapServer/50`).
- Fields: agency, transaction date, amount, vendor name, vendor state, merchant category description. **No cardholder ID** (rules work at agency level) and **no fraud labels**.
- Limits from DC Public Schools purchase-card guidance: $5,000 per purchase for goods, $2,500 for services.
- December 2024 has almost no transactions in the published data (data gap).

## Method

| Step | What | Tools |
|---|---|---|
| Categorise | 348 merchant category descriptions → 13 expense categories (keyword rules) | regex, pandas |
| ML 1: vendor-name classifier | TF-IDF character 2–4-grams + Logistic Regression; **GroupShuffleSplit by vendor** so test vendors are unseen | scikit-learn |
| Policy engine | 7 rules + 9 SQL queries (CTEs, CASE, UNION ALL, SUM OVER, RANK, moving average) | pandas, SQLite |
| ML 2: anomaly detection | Isolation Forest on 7 features (amount vs category median, vendor/category rarity within agency, out-of-state, …) vs z-score baseline, scored on weak labels | scikit-learn |
| Forensic test | Benford's law first digit, Nigrini MAD thresholds, per agency | NumPy |
| Review queue | risk score = rule points + 4 × anomaly percentile; reasons generated per row | pandas |

**Honest result:** Isolation Forest only modestly beats random (PR-AUC 0.196 vs 0.152) on rule-confirmed problems, and a simple z-score does better in the top 500. The rules carry most of the detection; the vendor-name model is the stronger ML piece.

## Files

| File | Purpose |
|---|---|
| `queries.sql` | SQL policy engine and analysis, commented |
| `analysis.py` | clean → categorise → text model → rules → SQL → anomaly model → Benford → review queue → `results.json` |
| `dashboard_template.html` → `index.html` | Chart.js dashboard (`python ../build.py expense-copilot`) |
| `scored_purchases.csv` | every purchase with its rule flags and risk score (created by `analysis.py`) |
| `powerbi_export.py` | CSVs for Power BI / Tableau |

## Run it

```bash
pip install pandas numpy scikit-learn
python analysis.py          # ~15 seconds
python ../build.py expense-copilot
```

*Independent project on public data; not affiliated with or endorsed by American Express or the DC government.*
