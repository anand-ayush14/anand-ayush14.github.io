# Credit Card Complaint Intelligence: American Express vs peers

**Live dashboard:** https://anand-ayush14.github.io/projects/complaint-intelligence/

American Express made *"reimagine customer experiences"* a strategic priority in 2026. This project uses the US regulator's public complaint database to answer:

1. What do Amex card members complain about, and where does Amex differ from Chase, Capital One and Citi?
2. Which problems are rising, and can we catch spikes early?
3. Can we predict which complaints will end in a fix or refund, so they are handled first?

## Key findings (2023-09 to 2026-10, 92,757 complaints, 14,883 about Amex)

| Finding | Number |
|---|---|
| Promotion / bonus-offer complaints, Amex share vs peers | **9.4% vs 5.3% (1.8×)** |
| Amex disputed-purchase complaints, last 12 months vs prior 12 | **+74%** (1,160 → 2,013) |
| Late responses to the CFPB (15-day rule) that came from Amex | **82 of 85 (96.5%)**, mostly Nov 2025 – Jan 2026 |
| Amex complaints closed with relief | 22.6% (Chase and Capital One similar; Citi ~50%) |
| Spike alerts flagged by the detector | 28 |
| Relief cases found in the top 20% of model scores | 33.0% (≈1.6× random) |

**Recommendations:** an in-app welcome-offer/promotion tracker; extra dispute staffing before the holidays, driven by the spike alert; zero tolerance on late regulatory responses; route high-score complaints to a remediation team; clearer application-denial explanations.

## Data

- Source: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/), public and free. Exported through its search API: product = Credit card; companies American Express, JPMorgan Chase, Capital One, Citibank.
- The export starts mid-August 2023; partial months and complaints still "In progress" are dropped.
- The CFPB no longer includes complaint narratives in the export, so the analysis uses the structured fields: issue, sub-issue, card type, state, channel, tags, response, timeliness.
- Counts are **not** adjusted for each issuer's number of card members: compare *mix and rates*, not raw volumes.

## Method

| Step | What | Tools |
|---|---|---|
| Clean | short issuer/issue names, relief flag, month, hours to reach company | pandas |
| SQL | 13 queries: issuer overview, issue mix and index vs peers, monthly trend with LAG, YoY growth, root-cause sub-issues with RANK, relief by issuer, late responses, vulnerable customers, 3-month moving average | SQLite (CTEs, CASE, window functions) |
| Spike detector | z-score of each issue-month vs that issue's previous 6 months; alert if z ≥ 2 and ≥ 20 complaints | pandas rolling |
| Model | Logistic Regression (one-hot), Random Forest, Gradient Boosting (native categorical); TimeSeriesSplit CV; train Sep 2023 – Mar 2026, test Apr – Oct 2026 | scikit-learn |
| Explain | permutation importance, cumulative gains | scikit-learn |

Model results (test set): champion **Gradient Boosting**, PR-AUC 0.422 (base rate 0.31), ROC-AUC 0.665. Within Amex alone the ROC-AUC is 0.591: the public fields mainly tell you *which issuer* and *which sub-issue*, so internal data (complaint text, transactions, history) would be needed for a strong model. This is stated openly on the dashboard.

## Files

| File | Purpose |
|---|---|
| `queries.sql` | all SQL, commented |
| `analysis.py` | cleaning → SQL → spike detector → models → `results.json` |
| `dashboard_template.html` → `index.html` | Chart.js dashboard (`python ../build.py complaint-intelligence`) |
| `powerbi_export.py` | CSVs for a Power BI / Tableau version |
| `data/` | the CFPB export (download instructions below) |

## Run it

```bash
pip install pandas numpy scikit-learn
python analysis.py          # ~1 minute
python ../build.py complaint-intelligence
```

To refresh the data, open this URL in a browser (one per company, change `company=`), and combine the CSVs:
`https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?company=AMERICAN%20EXPRESS%20COMPANY&product=Credit%20card&format=csv&size=40000&no_aggs=true`

*Independent project on public data; not affiliated with or endorsed by American Express.*
