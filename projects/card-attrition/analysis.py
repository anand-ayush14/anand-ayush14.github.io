"""
Credit Card Customer Attrition
==============================
Business question: about 1 in 6 card holders closed their card. Who is about to leave,
what are the early-warning signs, and is a retention campaign worth paying for?

Data: 10,127 credit card customers of a consumer bank (12 months of activity).
Target: Attrition_Flag ("Attrited Customer" = closed the card). Amounts in USD.

Steps
  1. Load and clean (rename columns, drop helper columns, check missing/unknown values)
  2. Answer business questions with SQL (SQLite in memory, see queries.sql)
  3. Feature engineering (average transaction value, spend per product, etc.)
  4. Compare 3 ML algorithms (5-fold stratified CV): Logistic Regression, Random Forest,
     Gradient Boosting. Champion explained with permutation importance.
  5. K-Means customer segmentation (k chosen by silhouette score)
  6. Retention campaign economics: who to target and the expected profit
  7. Export results for the dashboard (results.json)

Run:  python analysis.py
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA_URL = "https://raw.githubusercontent.com/trooulala/BankChurners/main/BankChurners.csv"

# Campaign assumptions (stated openly on the dashboard)
OFFER_COST = 50          # $ per customer contacted (fee waiver / cashback + agent time)
SAVE_RATE = 0.30         # share of true would-be leavers who stay because of the offer
INTERCHANGE = 0.02       # issuer earns ~2% of card spend
APR = 0.18               # interest earned on revolving balances
YEARS = 2                # value horizon for a saved customer

# ---------------------------------------------------------------- 1. load + clean
local = HERE / "BankChurners.csv"
raw = pd.read_csv(local if local.exists() else DATA_URL)
raw = raw.loc[:, ~raw.columns.str.startswith(("Unnamed", "Naive_Bayes"))]
df = raw.rename(columns={
    "CLIENTNUM": "client_id", "Customer_Age": "age", "Dependent_count": "dependents",
    "Education_Level": "education", "Marital_Status": "marital_status",
    "Income_Category": "income_category", "Card_Category": "card_category",
    "Months_on_book": "tenure_months", "Total_Relationship_Count": "products_held",
    "Months_Inactive_12_mon": "months_inactive", "Contacts_Count_12_mon": "contacts_12m",
    "Credit_Limit": "credit_limit", "Total_Revolving_Bal": "revolving_bal",
    "Avg_Open_To_Buy": "open_to_buy", "Total_Amt_Chng_Q4_Q1": "amt_chng_q4_q1",
    "Total_Trans_Amt": "total_trans_amt", "Total_Trans_Ct": "total_trans_ct",
    "Total_Ct_Chng_Q4_Q1": "ct_chng_q4_q1", "Avg_Utilization_Ratio": "utilisation",
    "Gender": "gender"})
df["attrited"] = (df.pop("Attrition_Flag") == "Attrited Customer").astype(int)
assert df.client_id.is_unique and df.isna().sum().sum() == 0
unknown = {c: int((df[c] == "Unknown").sum()) for c in ["education", "marital_status", "income_category"]}
print(f"Rows: {len(df):,} | attrition rate: {df.attrited.mean():.1%} | 'Unknown' values: {unknown}")
# 'Unknown' is kept as its own category: it is a real answer ("not disclosed"), not an error.

# ---------------------------------------------------------------- 2. SQL analysis
con = sqlite3.connect(":memory:")
df.to_sql("customers", con, index=False)
queries = [q.strip() for q in (HERE / "queries.sql").read_text().split(";") if "SELECT" in q]
sql_results = []
for q in queries:
    title = next(l for l in q.splitlines() if l.startswith("-- Q")).lstrip("- ").strip()
    res = pd.read_sql(q, con)
    sql_results.append((title, res))
    print(f"\n{title}\n{res.to_string(index=False)}")

# ---------------------------------------------------------------- 3. features
df["avg_txn_value"] = df.total_trans_amt / df.total_trans_ct
df["spend_per_product"] = df.total_trans_amt / df.products_held
# Behavioural features. Gender, education and marital status are left out: they add little
# and the bank should not target or deny offers based on personal attributes.
# open_to_buy is dropped (= credit_limit - revolving_bal, perfectly collinear).
features = ["age", "dependents", "tenure_months", "products_held", "months_inactive",
            "contacts_12m", "credit_limit", "revolving_bal", "utilisation",
            "total_trans_amt", "total_trans_ct", "amt_chng_q4_q1", "ct_chng_q4_q1",
            "avg_txn_value"]

# ---------------------------------------------------------------- 4. models
X, y = df[features], df["attrited"]
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, df.index, test_size=0.25, stratify=y, random_state=42)
candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                            class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                        random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
comparison, scores = [], {}
for name, est in candidates.items():
    r = cross_validate(est, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"])
    p = est.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    scores[name] = p
    comparison.append({"model": name,
                       "cv_auc": round(r["test_roc_auc"].mean(), 3),
                       "cv_pr_auc": round(r["test_average_precision"].mean(), 3),
                       "test_auc": round(roc_auc_score(y_test, p), 3),
                       "test_pr_auc": round(average_precision_score(y_test, p), 3)})
print("\nModel comparison:\n", pd.DataFrame(comparison).to_string(index=False))
champion = max(comparison, key=lambda c: c["cv_pr_auc"])["model"]       # chosen on CV only
p_champ = scores[champion]
print(f"Champion: {champion}")
print(classification_report(y_test, (p_champ >= 0.5).astype(int), digits=3))

imp = permutation_importance(candidates[champion], X_test, y_test, scoring="average_precision",
                             n_repeats=10, random_state=42, n_jobs=-1)
importance = pd.Series(imp.importances_mean, index=features).sort_values(ascending=False).head(8)
print("Top drivers (drop in PR-AUC when shuffled):\n", importance.round(3))

# ---------------------------------------------------------------- 5. segmentation
seg_cols = ["total_trans_amt", "total_trans_ct", "ct_chng_q4_q1", "revolving_bal",
            "utilisation", "products_held", "months_inactive", "contacts_12m"]
Z = StandardScaler().fit_transform(df[seg_cols])
sil = {}
for k in range(3, 7):
    lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)
    sil[k] = silhouette_score(Z, lab, sample_size=5000, random_state=42)
best_k = max(sil, key=sil.get)
# Business override: k=4 is used if its silhouette is within 0.02 of the best (easier to act on)
k = 4 if sil[4] >= sil[best_k] - 0.02 else best_k
df["segment"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)
print("\nSilhouette by k:", {kk: round(v, 3) for kk, v in sil.items()}, "-> k =", k)
seg = (df.groupby("segment")
         .agg(customers=("client_id", "size"), attrition=("attrited", "mean"),
              spend=("total_trans_amt", "mean"), txns=("total_trans_ct", "mean"),
              trend=("ct_chng_q4_q1", "mean"), revolving=("revolving_bal", "mean"),
              utilisation=("utilisation", "mean"), products=("products_held", "mean"),
              inactive=("months_inactive", "mean"))
         .reset_index())

def seg_name(r):
    if r.spend > 8000:
        return "High spenders"
    if r.attrition > 0.20:
        return "Fading users"
    if r.revolving > 1500 or r.utilisation > 0.45:
        return "Revolvers"
    return "Everyday users"

seg["name"] = seg.apply(seg_name, axis=1)
# make names unique if two segments share a rule
seg["name"] = seg.groupby("name")["name"].transform(
    lambda s: s if len(s) == 1 else [f"{v} {i + 1}" for i, v in enumerate(s)])
print(seg.round(2).to_string(index=False))

# ---------------------------------------------------------------- 6. campaign economics
t = df.loc[id_test, ["client_id", "attrited", "total_trans_amt", "revolving_bal"]].copy()
t["score"] = p_champ
t["value"] = YEARS * (INTERCHANGE * t.total_trans_amt + APR * t.revolving_bal)   # $ value if kept
t = t.sort_values("score", ascending=False).reset_index(drop=True)
t["n"] = np.arange(1, len(t) + 1)
t["gain"] = (t.attrited * t.value * SAVE_RATE).cumsum()
t["profit"] = t.gain - t.n * OFFER_COST
best = t.loc[t.profit.idxmax()]
total_leavers = t.attrited.sum()
campaign = {"offer_cost": OFFER_COST, "save_rate": SAVE_RATE, "years": YEARS,
            "best_n": int(best.n), "best_pct": round(100 * best.n / len(t), 1),
            "leavers_reached": int(t.head(int(best.n)).attrited.sum()),
            "leavers_reached_pct": round(100 * t.head(int(best.n)).attrited.sum() / total_leavers, 1),
            "profit": round(float(best.profit), 0),
            "spend": round(float(best.n * OFFER_COST), 0),
            "blanket_profit": round(float(t.profit.iloc[-1]), 0),
            "test_customers": int(len(t)), "avg_value_leaver": round(float(t.loc[t.attrited == 1, "value"].mean()), 0)}
pts = sorted(set(np.linspace(10, len(t), 50).astype(int)))
profit_curve = [{"pct": round(100 * n / len(t), 1), "profit": round(float(t.profit.iloc[n - 1]), 0)} for n in pts]
gains = [{"top_pct": p, "captured": round(100 * t.head(int(len(t) * p / 100)).attrited.sum() / total_leavers, 1)}
         for p in (5, 10, 15, 20, 30, 40, 50, 100)]
print("\nCampaign:", campaign)
print("Gains:", gains)

# ---------------------------------------------------------------- 7. export
out = {
    "kpis": {"customers": int(len(df)), "attrition_rate": round(float(df.attrited.mean()) * 100, 1),
             "attrited": int(df.attrited.sum()), "champion": champion,
             "champion_auc": next(c["test_auc"] for c in comparison if c["model"] == champion),
             "unknown": unknown, "k": int(k)},
    "sql": {t_.split(".")[0]: r.to_dict(orient="records") for t_, r in sql_results},
    "models": comparison,
    "importance": {k_: round(float(v), 4) for k_, v in importance.items()},
    "silhouette": {str(kk): round(float(v), 3) for kk, v in sil.items()},
    "segments": [{"name": r.name, "customers": int(r.customers), "attrition": round(100 * r.attrition, 1),
                  "spend": int(r.spend), "txns": round(r.txns, 1), "trend": round(r.trend, 2),
                  "revolving": int(r.revolving), "utilisation": round(100 * r.utilisation, 1),
                  "products": round(r.products, 1), "inactive": round(r.inactive, 1)}
                 for r in seg.itertuples()],
    "campaign": campaign, "profit_curve": profit_curve, "gains": gains,
}
(HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))
print("\nSaved results.json")
