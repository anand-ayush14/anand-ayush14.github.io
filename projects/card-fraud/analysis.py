"""
Card Transaction Fraud Detection
================================
Business question: fraud is only ~0.17% of card transactions, so a fraud team cannot review
everything. Which transactions should analysts review first, and how much fraud does a
fixed daily review budget actually stop?

Data: 284,807 European card transactions over 2 days (Sep 2013), 492 frauds.
V1-V28 are PCA components (the bank anonymised the raw fields for confidentiality),
plus Time (seconds since first transaction), Amount (EUR) and Class (1 = fraud).
Source: Worldline & ULB Machine Learning Group (Dal Pozzolo et al., 2015).

Steps
  1. Load, clean, add hour/day features
  2. Answer business questions with SQL (SQLite in memory, see queries.sql)
  3. TIME-BASED split: train on the earlier 75% of transactions, test on the later 25%
     (a fraud model is always used on future transactions, so a random split would be optimistic)
  4. Compare 4 algorithms on PR-AUC (right metric for rare events):
       Isolation Forest (unsupervised anomaly detection, needs no fraud labels)
       Logistic Regression, Random Forest, Gradient Boosting (supervised)
  5. Turn the champion model into an operating policy: alert budget, euro savings,
     cost-based threshold, and which signals drive alerts (permutation importance)
  6. Export results for the dashboard (results.json)

Run:  python analysis.py
"""
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA_URL = ("https://raw.githubusercontent.com/nsethi31/"
            "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv")
REVIEW_COST_EUR = 5      # assumption: analyst time + customer contact per alert reviewed

# ---------------------------------------------------------------- 1. load + clean
local = HERE / "creditcard.csv"
df = pd.read_csv(local if local.exists() else DATA_URL)
assert df.isna().sum().sum() == 0
dupes = df.duplicated().sum()
df = df.drop_duplicates().sort_values("Time").reset_index(drop=True)   # exact duplicate rows
df = df.rename(columns={"Time": "time_sec", "Amount": "amount", "Class": "is_fraud"})
df.insert(0, "txn_id", np.arange(1, len(df) + 1))
df["day"] = (df["time_sec"] // 86400).astype(int) + 1
df["hour"] = ((df["time_sec"] // 3600) % 24).astype(int)   # hours since start, mod 24
df["log_amount"] = np.log1p(df["amount"])
print(f"Removed {dupes:,} duplicate rows | {len(df):,} transactions | "
      f"{df.is_fraud.sum()} frauds ({df.is_fraud.mean():.3%})")

# ---------------------------------------------------------------- 2. SQL analysis
con = sqlite3.connect(":memory:")
df.to_sql("transactions", con, index=False)
queries = [q.strip() for q in (HERE / "queries.sql").read_text().split(";") if "SELECT" in q]
sql_results = []
for q in queries:
    title = next(l for l in q.splitlines() if l.startswith("-- Q")).lstrip("- ").strip()
    res = pd.read_sql(q, con)
    sql_results.append((title, res))
    print(f"\n{title}\n{res.head(12).to_string(index=False)}")

# ---------------------------------------------------------------- 3. time-based split
features = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour"]
cut = int(len(df) * 0.75)
train, test = df.iloc[:cut], df.iloc[cut:]
X_train, y_train = train[features], train["is_fraud"]
X_test, y_test = test[features], test["is_fraud"]
test_hours = (test.time_sec.max() - test.time_sec.min()) / 3600
print(f"\nTrain: {len(train):,} txns ({y_train.sum()} frauds) | "
      f"Test: {len(test):,} txns ({y_test.sum()} frauds) over the last {test_hours:.1f} hours")

# ---------------------------------------------------------------- 4. model comparison
candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features="sqrt",
                                            class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                        max_leaf_nodes=31, l2_regularization=1.0,
                                                        random_state=42),
}
# Time-series CV: each fold trains on the past and validates on the next block of time
cv = TimeSeriesSplit(n_splits=3)
comparison, scores = [], {}
for name, est in candidates.items():
    t0 = time.time()
    cv_ap = cross_val_score(est, X_train, y_train, cv=cv, scoring="average_precision")
    p = est.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    scores[name] = p
    comparison.append({"model": name, "type": "Supervised",
                       "cv_pr_auc": round(cv_ap.mean(), 3), "cv_std": round(cv_ap.std(), 3),
                       "test_pr_auc": round(average_precision_score(y_test, p), 3),
                       "test_roc_auc": round(roc_auc_score(y_test, p), 3)})
    print(f"{name}: CV PR-AUC {cv_ap.mean():.3f} | test PR-AUC {comparison[-1]['test_pr_auc']} "
          f"({time.time() - t0:.0f}s)")

# Isolation Forest: trained WITHOUT labels. Shows how far pure anomaly detection gets,
# which matters for brand-new fraud patterns that have no labelled history yet.
iso = make_pipeline(StandardScaler(), IsolationForest(n_estimators=300, contamination="auto",
                                                      random_state=42, n_jobs=-1))
iso.fit(X_train)
p_iso = -iso.score_samples(X_test)            # higher = more anomalous
scores["Isolation Forest"] = p_iso
comparison.append({"model": "Isolation Forest", "type": "Unsupervised", "cv_pr_auc": None,
                   "cv_std": None, "test_pr_auc": round(average_precision_score(y_test, p_iso), 3),
                   "test_roc_auc": round(roc_auc_score(y_test, p_iso), 3)})
print("\nModel comparison:\n", pd.DataFrame(comparison).to_string(index=False))

# Champion picked on CV PR-AUC (never on the test set)
sup = [c for c in comparison if c["type"] == "Supervised"]
champion = max(sup, key=lambda c: c["cv_pr_auc"])["model"]
p_champ = scores[champion]
print(f"Champion: {champion}")

# ---------------------------------------------------------------- 5. operating policy
t = test[["txn_id", "amount", "is_fraud"]].copy()
t["score"] = p_champ
t = t.sort_values("score", ascending=False).reset_index(drop=True)
n_fraud, fraud_eur = t.is_fraud.sum(), t.loc[t.is_fraud == 1, "amount"].sum()

def budget_row(n_alerts, label):
    top = t.head(n_alerts)
    caught = int(top.is_fraud.sum())
    return {"label": label, "alerts": int(n_alerts),
            "alerts_per_day": int(round(n_alerts / test_hours * 24)),
            "frauds_caught": caught, "recall": round(100 * caught / n_fraud, 1),
            "precision": round(100 * caught / n_alerts, 1),
            "eur_caught_pct": round(100 * top.loc[top.is_fraud == 1, "amount"].sum() / fraud_eur, 1)}

budget = [budget_row(int(len(t) * s), f"Top {100 * s:g}%")
          for s in (0.0005, 0.001, 0.002, 0.005, 0.01)]
print("\nAlert budget (champion, test set):\n", pd.DataFrame(budget).to_string(index=False))

# Same budget comparison for Isolation Forest (top 0.2%)
t_iso = test[["is_fraud"]].assign(score=p_iso).sort_values("score", ascending=False)
n02 = int(len(t) * 0.002)
iso_recall_02 = round(100 * t_iso.head(n02).is_fraud.sum() / n_fraud, 1)

# Cost-based cut-off: missed fraud costs its amount, every alert costs REVIEW_COST_EUR
t["cum_alerts"] = np.arange(1, len(t) + 1)
t["cum_fraud_eur"] = (t.amount * t.is_fraud).cumsum()
t["total_cost"] = t.cum_alerts * REVIEW_COST_EUR + (fraud_eur - t.cum_fraud_eur)
best = t.loc[t.total_cost.idxmin()]
no_model_cost = fraud_eur                                     # review nothing: lose all fraud
cost = {"review_cost_eur": REVIEW_COST_EUR, "best_alerts": int(best.cum_alerts),
        "best_alerts_pct": round(100 * best.cum_alerts / len(t), 2),
        "fraud_eur_test": round(float(fraud_eur), 0),
        "eur_prevented": round(float(best.cum_fraud_eur), 0),
        "review_spend": round(float(best.cum_alerts * REVIEW_COST_EUR), 0),
        "net_saving": round(float(no_model_cost - best.total_cost), 0)}
curve_pts = np.unique(np.geomspace(1, 5000, 40).astype(int))
cost_curve = [{"alerts": int(n), "total_cost": round(float(t.total_cost.iloc[n - 1]), 0)} for n in curve_pts]
print("\nCost-optimal policy:", cost)

# Precision-recall curves (downsampled for the chart)
pr_curves = {}
for name in [champion, "Logistic Regression", "Isolation Forest"]:
    pr, rc, _ = precision_recall_curve(y_test, scores[name])
    # interpolated precision: best precision achievable at each recall level (2% steps)
    grid = np.linspace(0, 1, 51)
    pr_curves[name] = [{"r": round(float(g), 3), "p": round(float(pr[rc >= g].max()), 3)} for g in grid]

# Which signals drive the champion? Permutation importance on PR-AUC
# (sample: all test frauds + 20,000 genuine transactions to keep it fast)
samp = pd.concat([test[test.is_fraud == 1], test[test.is_fraud == 0].sample(20000, random_state=1)])
imp = permutation_importance(candidates[champion], samp[features], samp["is_fraud"],
                             scoring="average_precision", n_repeats=5, random_state=42, n_jobs=-1)
importance = (pd.Series(imp.importances_mean, index=features).sort_values(ascending=False).head(8))
print("\nTop drivers (drop in PR-AUC when shuffled):\n", importance.round(3))

# Fraud vs genuine profile for the top 3 drivers (shows direction of effect)
profile = []
for f in importance.index[:3]:
    profile.append({"feature": f, "fraud_median": round(float(df.loc[df.is_fraud == 1, f].median()), 2),
                    "genuine_median": round(float(df.loc[df.is_fraud == 0, f].median()), 2)})

# ---------------------------------------------------------------- 6. export
out = {
    "kpis": {"transactions": int(len(df)), "frauds": int(df.is_fraud.sum()),
             "fraud_rate": round(float(df.is_fraud.mean()) * 100, 3),
             "duplicates_removed": int(dupes), "test_txns": int(len(test)),
             "test_frauds": int(n_fraud), "test_hours": round(float(test_hours), 1),
             "champion": champion, "iso_recall_02": iso_recall_02},
    "sql": {t_.split(".")[0]: r.to_dict(orient="records") for t_, r in sql_results},
    "models": comparison, "budget": budget, "cost": cost, "cost_curve": cost_curve,
    "pr_curves": pr_curves,
    "importance": {k: round(float(v), 4) for k, v in importance.items()},
    "profile": profile,
}
(HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))
print("\nSaved results.json")
