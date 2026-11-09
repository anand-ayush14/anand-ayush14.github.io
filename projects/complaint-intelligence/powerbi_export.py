"""Export tidy CSVs for Power BI / Tableau: complaints fact table + monthly issue counts."""
from pathlib import Path
import pandas as pd
HERE = Path(__file__).parent
exec(open(HERE / "analysis.py").read().split("# ---------------------------------------------------------------- 2. SQL")[0])
out = HERE / "powerbi"; out.mkdir(exist_ok=True)
df[["complaint_id", "issuer", "month", "issue_short", "sub_issue", "sub_product", "state", "channel",
    "tags", "response", "relief", "monetary", "timely"]].to_csv(out / "complaints.csv", index=False)
(df.groupby(["issuer", "month", "issue_short"]).size().rename("complaints").reset_index()
   .to_csv(out / "monthly_issue_counts.csv", index=False))
print("Wrote", list(out.iterdir()))
