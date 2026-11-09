"""
Credit Card Default Risk Analysis
=================================
Business question: which credit card customers are most likely to miss next month's
payment, and what should the card issuer do about it?

Data: UCI "Default of Credit Card Clients" - 30,000 card holders in Taiwan (Apr-Sep 2005).
Amounts are in NT$. Target column: defaulted (1 = missed the next payment).

Steps
  1. Load and clean the data
  2. Answer business questions with SQL (SQLite in memory, see queries.sql)
  3. Feature engineering (utilisation, payment behaviour)
  4. Compare 3 ML algorithms (5-fold CV): Gradient Boosting = champion (risk tiers),
     Logistic Regression = explainer (which factors drive risk)
  5. Export results for the dashboard (results.json)

Run:  python analysis.py
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA_URL = ("https://raw.githubusercontent.com/Tvarnex/AppliedMachineLearning/"
            "0ae785aa4971cd3ca1848935a61798fcc9464cf6/CreditData/UCI_Credit_Card.csv")

# ---------------------------------------------------------------- 1. load + clean
local = HERE / "UCI_Credit_Card.csv"
df = pd.read_csv(local if local.exists() else DATA_URL)
df = df.rename(columns=str.lower).rename(columns={"default.payment.next.month": "defaulted"})

# Undocumented codes: education 0,5,6 -> 4 ("other"); marriage 0 -> 3 ("other")
df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
df["marriage"] = df["marriage"].replace({0: 3})
assert df["id"].is_unique and df.isna().sum().sum() == 0
print(f"Rows: {len(df):,} | default rate: {df.defaulted.mean():.1%}")

# ---------------------------------------------------------------- 2. SQL analysis
con = sqlite3.connect(":memory:")
df.to_sql("clients", con, index=False)
queries = [q.strip() for q in (HERE / "queries.sql").read_text().split(";") if "SELECT" in q]
sql_results = []
for q in queries:
    title = next(l for l in q.splitlines() if l.startswith("-- Q")).lstrip("- ").strip()
    res = pd.read_sql(q, con)
    sql_results.append((title, res))
    print(f"\n{title}\n{res.to_string(index=False)}")

# ---------------------------------------------------------------- 3. features
pay_cols = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
bill_cols = [f"bill_amt{i}" for i in range(1, 7)]
amt_cols = [f"pay_amt{i}" for i in range(1, 7)]

df["utilisation"] = (df["bill_amt1"] / df["limit_bal"]).clip(-1, 3)
df["months_late_max"] = df[pay_cols].clip(lower=0).max(axis=1)
df["late_count_6m"] = (df[pay_cols] > 0).sum(axis=1)
total_bill = df[bill_cols].clip(lower=0).sum(axis=1)
df["pay_ratio_6m"] = (df[amt_cols].sum(axis=1) / total_bill.replace(0, np.nan)).fillna(1).clip(0, 2)

# Behavioural features only. Sex, marital status and education are deliberately EXCLUDED:
# fair-lending rules restrict using protected/demographic attributes in credit decisions.
# pay_ratio_6m is kept for EDA but left out of the model: it overlaps with the lateness
# features (multicollinearity) and flipped sign, which would make coefficients misleading.
features = ["limit_bal", "age", "pay_0", "months_late_max", "late_count_6m", "utilisation"]

# ---------------------------------------------------------------- 4. model
X, y = df[features], df["defaulted"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)          # split BEFORE scaling
# 4a. Compare algorithms with 5-fold stratified cross-validation on the TRAINING set
candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                            class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                        random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
comparison = []
for name, est in candidates.items():
    cv_auc = cross_val_score(est, X_train, y_train, cv=cv, scoring="roc_auc")
    test_auc = roc_auc_score(y_test, est.fit(X_train, y_train).predict_proba(X_test)[:, 1])
    comparison.append({"model": name, "cv_auc": round(cv_auc.mean(), 3),
                       "cv_std": round(cv_auc.std(), 3), "test_auc": round(test_auc, 3)})
print("\nModel comparison (ROC-AUC):\n", pd.DataFrame(comparison).to_string(index=False))

# 4b. Champion / explainer pairing (common in banking):
#     - Gradient Boosting = champion model: best ranking power -> used for risk tiers
#     - Logistic Regression = explainer: coefficients show WHICH factors drive risk
model = candidates["Gradient Boosting"]                         # already fitted on train
explainer = candidates["Logistic Regression"]                   # scaler fit on train only
proba_test = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba_test)
pay_ratio_band = pd.cut(df["pay_ratio_6m"], [-0.01, 0.05, 0.2, 0.5, 1.0, 2.0],
                        labels=["<5%", "5-20%", "20-50%", "50-100%", "100%+"])
pay_ratio_tbl = (df.groupby(pay_ratio_band, observed=False)["defaulted"]
                   .agg(["size", "mean"]).reset_index())
print("\nDefault rate by share of bills repaid (6 months):\n", pay_ratio_tbl.round(3).to_string(index=False))

print(f"\nTest ROC-AUC: {auc:.3f}")
# GB is not class-weighted, so 0.5 is too strict a cut-off for a 22% base rate
print(classification_report(y_test, (proba_test >= 0.3).astype(int), digits=3))

coefs = pd.Series(explainer[-1].coef_[0], index=features).sort_values(key=abs, ascending=False)
print("Standardised coefficients (bigger = stronger effect):\n", coefs.round(3))

# Risk tiers on the TEST set (honest, unseen data)
test = X_test.copy()
test["defaulted"], test["score"] = y_test.values, proba_test
# Rank-based tiers: bottom 60% = Low, next 20% = Medium, next 10% = High, top 10% = Very high
pct = test["score"].rank(pct=True)
test["tier"] = pd.cut(pct, [0, 0.6, 0.8, 0.9, 1.0],
                      labels=["Low", "Medium", "High", "Very high"], include_lowest=True)
tiers = (test.groupby("tier", observed=False)
             .agg(customers=("score", "size"), default_rate=("defaulted", "mean"))
             .reset_index())
tiers["share"] = tiers["customers"] / tiers["customers"].sum()
print("\nRisk tiers (test set):\n", tiers.round(3).to_string(index=False))

# Precision/recall if we act on High + Very high
flag = test["tier"].isin(["High", "Very high"])
caught = test.loc[flag, "defaulted"].sum() / test["defaulted"].sum()
precision = test.loc[flag, "defaulted"].mean()
print(f"Acting on High+Very high: flags {flag.mean():.1%} of customers, "
      f"catches {caught:.1%} of defaulters, precision {precision:.1%}")

# ---------------------------------------------------------------- 5. export
def rows(res):
    return res.to_dict(orient="records")

out = {
    "kpis": {
        "customers": int(len(df)),
        "default_rate": round(float(df.defaulted.mean()) * 100, 1),
        "avg_limit": int(df.limit_bal.mean()),
        "auc": round(float(auc), 3),
        "flag_share": round(float(flag.mean()) * 100, 1),
        "caught": round(float(caught) * 100, 1),
        "precision": round(float(precision) * 100, 1),
    },
    "sql": {t.split(".")[0]: rows(r) for t, r in sql_results},
    "models": comparison,
    "pay_ratio": [{"band": str(r.pay_ratio_6m), "customers": int(r.size),
                   "default_rate": round(float(r.mean) * 100, 1)} for r in pay_ratio_tbl.itertuples()],
    "coefs": {k: round(float(v), 3) for k, v in coefs.items()},
    "tiers": [{"tier": str(r.tier), "customers": int(r.customers),
               "default_rate": round(float(r.default_rate) * 100, 1),
               "share": round(float(r.share) * 100, 1)} for r in tiers.itertuples()],
}
(HERE / "results.json").write_text(json.dumps(out, indent=2))
print("\nSaved results.json")
