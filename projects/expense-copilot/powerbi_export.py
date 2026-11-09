"""Export tidy CSVs for Power BI / Tableau (run analysis.py first)."""
from pathlib import Path
import pandas as pd
HERE = Path(__file__).parent
out = HERE / "powerbi"; out.mkdir(exist_ok=True)
s = pd.read_csv(HERE / "scored_purchases.csv", parse_dates=["date"])
s.to_csv(out / "purchases_scored.csv", index=False)
s.assign(month=s.date.dt.strftime("%Y-%m")).groupby(["month", "agency", "category"]).agg(
    purchases=("amount", "size"), spend=("amount", "sum")).reset_index().to_csv(out / "monthly_agency_category.csv", index=False)
print("Wrote", sorted(p.name for p in out.iterdir()))
