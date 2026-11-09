"""
Credit Card Complaint Intelligence (American Express vs peers)
==============================================================
Business question: American Express made "reimagine customer experiences" a strategic
priority in 2026. What do card members complain about, where does Amex do worse than
its peers, which issues are rising, and which complaints will need a fix or refund?

Data: every CFPB consumer complaint about a credit card filed against American Express,
JPMorgan Chase, Capital One and Citibank from Aug 2023 to Oct 2026 (public, free,
https://www.consumerfinance.gov/data-research/consumer-complaints/). About 94,000 rows.
Note: the CFPB no longer exposes complaint narratives in its export, so this project
works on the structured fields (issue, sub-issue, channel, state, tags, outcome).

Steps
  1. Load and clean (short issuer and issue names, relief flag, month)
  2. Answer business questions with SQL (SQLite in memory, see queries.sql)
  3. Spike detector: flag Amex issue-months more than 2 standard deviations above the
     issue's own 6-month baseline
  4. Predict which complaints end in relief (monetary or non-monetary) with 3 algorithms,
     time-based split (train on older months, test on the latest 7 months)
  5. Priority routing: how many relief cases sit in the top 20% of model scores
  6. Export results for the dashboard (results.json)

Run:  python analysis.py
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

HERE = Path(__file__).parent
DATA = HERE / "data" / "cfpb_cc_complaints_2022_2026.csv"

ISSUERS = {"AMERICAN EXPRESS COMPANY": "Amex", "JPMORGAN CHASE & CO.": "Chase",
           "CAPITAL ONE FINANCIAL CORPORATION": "Capital One", "CITIBANK, N.A.": "Citi"}
ISSUE_SHORT = {
    "Problem with a purchase shown on your statement": "Disputed purchase",
    "Other features, terms, or problems": "Other terms / problems",
    "Getting a credit card": "Getting a card",
    "Fees or interest": "Fees or interest",
    "Closing your account": "Account closure",
    "Advertising and marketing, including promotional offers": "Promotions / bonus offers",
    "Problem when making payments": "Payment problems",
    "Trouble using your card": "Trouble using card",
    "Incorrect information on your report": "Credit report errors",
    "Problem with a company's investigation into an existing problem": "Investigation problems",
    "Struggling to pay your bill": "Struggling to pay",
    "Improper use of your report": "Improper report use",
    "Problem with a credit reporting company's investigation into an existing problem": "Investigation problems",
    "Credit monitoring or identity theft protection services": "Credit monitoring",
    "Unable to get your credit report or credit score": "Can't get credit report",
}

# ---------------------------------------------------------------- 1. load + clean
def download():
    """Fetch the CFPB export (credit-card complaints for the 4 issuers) if it is not on disk yet."""
    import io
    import urllib.parse
    import urllib.request
    api = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
    parts = []
    for company in ISSUERS:
        windows = [("2022-01-01", "2024-01-01"), ("2024-01-01", "2030-01-01")]  # keep each request < 40,000 rows
        for lo, hi in windows:
            q = urllib.parse.urlencode({"company": company, "product": "Credit card", "date_received_min": lo,
                                        "date_received_max": hi, "format": "csv", "size": 40000, "no_aggs": "true"})
            with urllib.request.urlopen(api + "?" + q, timeout=300) as r:
                parts.append(pd.read_csv(io.BytesIO(r.read()), dtype=str))
            print("downloaded", company, lo, len(parts[-1]))
    DATA.parent.mkdir(exist_ok=True)
    pd.concat(parts).drop_duplicates("Complaint ID").to_csv(DATA, index=False)


if not DATA.exists():
    download()
raw = pd.read_csv(DATA, dtype=str)
df = raw.rename(columns={
    "Date received": "date_received", "Sub-product": "sub_product", "Issue": "issue",
    "Sub-issue": "sub_issue", "Company": "company", "State": "state", "Tags": "tags",
    "Submitted via": "channel", "Date sent to company": "date_sent",
    "Company response to consumer": "response", "Timely response?": "timely_raw",
    "Complaint ID": "complaint_id"})
df = df.drop_duplicates("complaint_id")
df = df.replace({"None": np.nan})
df["issuer"] = df.company.map(ISSUERS)
df["issue_short"] = df.issue.map(ISSUE_SHORT).fillna(df.issue)
received = pd.to_datetime(df.date_received, utc=True)
df["month"] = received.dt.strftime("%Y-%m")
df["hours_to_company"] = ((pd.to_datetime(df.date_sent, utc=True) - received)
                          .dt.total_seconds() / 3600).clip(0, 24 * 30)
df["relief"] = df.response.isin(["Closed with monetary relief", "Closed with non-monetary relief"]).astype(int)
df["monetary"] = (df.response == "Closed with monetary relief").astype(int)
df["timely"] = (df.timely_raw == "Yes").astype(int)
# keep finished complaints and full months only (the export starts mid-Aug 2023)
df = df[(df.response != "In progress") & (df.month >= "2023-09") & (df.month <= "2026-10")].copy()
print(f"{len(df):,} closed complaints, {df.month.min()} to {df.month.max()}")
print(df.issuer.value_counts().to_string())

# ---------------------------------------------------------------- 2. SQL
con = sqlite3.connect(":memory:")
df[["complaint_id", "issuer", "month", "issue", "issue_short", "sub_issue", "sub_product", "state",
    "channel", "tags", "relief", "monetary", "timely"]].to_sql("complaints", con, index=False)
queries = [q.strip() for q in (HERE / "queries.sql").read_text().split(";") if "SELECT" in q]
sql = {}
for q in queries:
    title = next(l for l in q.splitlines() if l.startswith("-- Q")).lstrip("- ").strip()
    res = pd.read_sql(q, con)
    sql[title.split(".")[0]] = res  # key like "Q7b"
    print(f"\n{title}\n{res.head(15).to_string(index=False)}")

# ---------------------------------------------------------------- 3. spike detector
amex = df[df.issuer == "Amex"]
im = amex.pivot_table(index="month", columns="issue_short", values="complaint_id",
                      aggfunc="count", fill_value=0).sort_index()
base_mean = im.shift(1).rolling(6, min_periods=6).mean()
base_std = im.shift(1).rolling(6, min_periods=6).std()
z = (im - base_mean) / base_std.replace(0, np.nan)
spikes = []
for month in z.index:
    for issue in z.columns:
        zv, n = z.loc[month, issue], im.loc[month, issue]
        if pd.notna(zv) and zv >= 2 and n >= 20:
            spikes.append({"month": month, "issue": issue, "complaints": int(n),
                           "baseline": round(float(base_mean.loc[month, issue]), 1), "z": round(float(zv), 1)})
spikes = sorted(spikes, key=lambda s: s["month"], reverse=True)
print(f"\nSpikes flagged: {len(spikes)}")
for s in spikes[:10]:
    print(s)
top_issues = amex.issue_short.value_counts().head(5).index.tolist()
spike_series = {i: im[i].astype(int).tolist() for i in top_issues}

# ---------------------------------------------------------------- 4. relief model
CAT = ["issuer", "issue", "sub_issue", "sub_product", "state", "channel"]
df["older_american"] = df.tags.fillna("").str.contains("Older").astype(int)
df["servicemember"] = df.tags.fillna("").str.contains("Service").astype(int)
df["month_of_year"] = df.month.str[5:].astype(int)
NUM = ["older_american", "servicemember", "hours_to_company", "month_of_year"]
X = df[CAT + NUM].copy()
X[CAT] = X[CAT].fillna("Unknown")
y = df.relief.values
order = np.argsort(df.month.values, kind="stable")
X, y, months = X.iloc[order].reset_index(drop=True), y[order], df.month.values[order]
test_mask = months >= "2026-04"
X_tr, X_te, y_tr, y_te = X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]
print(f"\nTrain {len(X_tr):,} (to 2026-03), test {len(X_te):,} (2026-04 to 2026-10); relief base rate {y.mean():.3f}")

onehot = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT)],
                           remainder="passthrough")
ordinal = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                    encoded_missing_value=-1), CAT)], remainder="passthrough")
candidates = {
    "Logistic Regression": make_pipeline(onehot, LogisticRegression(max_iter=2000, C=0.5)),
    "Random Forest": make_pipeline(ordinal, RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                                   n_jobs=-1, random_state=42)),
    "Gradient Boosting": make_pipeline(ordinal, HistGradientBoostingClassifier(
        categorical_features=list(range(len(CAT))), learning_rate=0.08, max_iter=300,
        max_leaf_nodes=31, random_state=42)),
}
tscv = TimeSeriesSplit(n_splits=4)
models, fitted = [], {}
for name, model in candidates.items():
    cv = cross_val_score(model, X_tr, y_tr, cv=tscv, scoring="average_precision")
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    fitted[name] = (model, p)
    models.append({"model": name, "cv_pr_auc": round(cv.mean(), 3),
                   "test_pr_auc": round(average_precision_score(y_te, p), 3),
                   "test_auc": round(roc_auc_score(y_te, p), 3)})
    print(models[-1])
champion = max(models, key=lambda m: m["test_pr_auc"])["model"]
model, p = fitted[champion]
pi = permutation_importance(model, X_te, y_te, scoring="average_precision", n_repeats=5,
                            random_state=42, n_jobs=-1)
importance = dict(sorted(zip(X.columns, pi.importances_mean.round(4)), key=lambda t: -t[1]))
print("Importance:", importance)

# ---------------------------------------------------------------- 5. priority routing
rank = np.argsort(-p)
cum = np.cumsum(y_te[rank])
gains = []
for pct in [5, 10, 20, 30, 40, 50, 60, 80, 100]:
    k = int(len(p) * pct / 100)
    gains.append({"top_pct": pct, "captured": round(100 * cum[k - 1] / y_te.sum(), 1),
                  "precision": round(100 * cum[k - 1] / k, 1)})
print("Gains:", gains)
g20 = next(g for g in gains if g["top_pct"] == 20)
amex_te = (X_te.issuer == "Amex").values
amex_auc = round(roc_auc_score(y_te[amex_te], p[amex_te]), 3)

# ---------------------------------------------------------------- 6. export
q1 = sql["Q1"].set_index("issuer")
amex_row = q1.loc["Amex"]
q5 = sql["Q5"]
results = {
    "kpis": {
        "complaints": int(len(df)), "amex": int(amex_row.complaints),
        "months": f"{df.month.min()} to {df.month.max()}",
        "amex_relief": float(amex_row.relief_pct),
        "peer_relief": round(100 * df[df.issuer != "Amex"].relief.mean(), 1),
        "base_rate": round(100 * y.mean(), 1), "champion": champion,
        "routing_top20": g20["captured"], "amex_test_auc": amex_auc,
        "spikes": len(spikes), "amex_yoy": round(100 * (int(q5.last_12m.sum()) - int(q5.prior_12m.sum()))
                                                 / int(q5.prior_12m.sum()), 0),
    },
    "sql": {k: v.replace({np.nan: None}).to_dict(orient="records") for k, v in sql.items()},
    "models": models, "importance": importance, "gains": gains,
    "spikes": spikes[:12], "spike_months": im.index.tolist(), "spike_series": spike_series,
}
(HERE / "results.json").write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("\nWrote results.json")
