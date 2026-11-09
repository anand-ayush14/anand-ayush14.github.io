"""Compute every tool's output for the 3 demo members (Python reference) and bundle data for the web demo."""
import json
from datetime import date
from pathlib import Path
import pandas as pd
import engine

HERE = Path(__file__).parent
AS_OF = date(2026, 11, 9)
tx = engine.load_transactions()
members = json.loads((HERE / "data" / "members.json").read_text())
ref = {m: {"unused": engine.get_unused_credits(tx, m, AS_OF), "ytd": engine.year_to_date(tx, m, AS_OF),
           "fee": engine.value_vs_fee(tx, m, AS_OF)} for m in members}
web = {"as_of": AS_OF.isoformat(), "cards": json.loads((HERE / "cards.json").read_text()),
       "members": members, "tx": json.loads(tx.assign(date=tx.date.dt.strftime("%Y-%m-%d")).to_json(orient="records")),
       "reference": ref}
(HERE / "results.json").write_text(json.dumps(web, indent=1))
for m in members:
    print(m, [(f["card"], f["credits_used_ytd"], f["credits_missed_ytd"], f["net_vs_fee"]) for f in ref[m]["fee"]],
          "open:", len(ref[m]["unused"]), "left $", round(sum(r["remaining"] for r in ref[m]["unused"]), 2))
