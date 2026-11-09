"""
Corporate Card Expense Copilot
==============================
Business question: American Express is launching an AI expense-management platform for
business customers (Center and Hyper acquisitions, 2025-26). Can we automatically
categorise company-card spend, catch policy breaches (split purchases, over-limit,
duplicates, sensitive merchants) and give a manager a short, ranked review queue?

Data: 125,615 real purchase-card transactions of the District of Columbia government,
Jun 2023 - Sep 2026 (DC Open Data, CC BY 4.0): agency, date, amount, vendor, vendor state,
merchant category (MCC). There is no cardholder ID and no fraud label.

Steps
  1. Load and clean (refunds separated, month, weekday)
  2. Auto-categorise spend into 13 expense categories from the merchant category
  3. Policy rules (over limit, just under limit, split purchase, duplicate, sensitive
     merchant, weekend, round amount) + SQL analysis (queries.sql)
  4. Anomaly detection: Isolation Forest vs a simple z-score baseline, evaluated on how
     well each surfaces rule-confirmed problems it never saw (weak labels)
  5. Benford's law first-digit test per agency (forensic accounting check)
  6. Risk score and a ranked review queue with plain-English reasons
  7. Export results for the dashboard (results.json)

Run:  python analysis.py
"""
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

HERE = Path(__file__).parent
DATA = HERE / "data" / "dc_purchase_card_2023_2026.csv"
GOODS_LIMIT, SERVICES_LIMIT = 5000, 2500

# ---------------------------------------------------------------- 1. load + clean
def download(since="2023-06-01"):
    """Fetch DC purchase-card transactions from the city's ArcGIS REST service if not on disk yet."""
    import urllib.parse
    import urllib.request
    layer = "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Public_Service_WebMercator/MapServer/50/query"
    def call(params):
        data = urllib.parse.urlencode({**params, "f": "json"}).encode()
        with urllib.request.urlopen(layer, data=data, timeout=120) as r:
            return json.load(r)
    ids = sorted(call({"where": f"TRANSACTION_DATE >= DATE '{since}'", "returnIdsOnly": "true"})["objectIds"])
    fields = "OBJECTID,AGENCY,TRANSACTION_DATE,TRANSACTION_AMOUNT,VENDOR_NAME,VENDOR_STATE_PROVINCE,MCC_DESCRIPTION"
    rows = []
    for i in range(0, len(ids), 1000):
        res = call({"objectIds": ",".join(map(str, ids[i:i + 1000])), "outFields": fields, "returnGeometry": "false"})
        rows += [f["attributes"] for f in res["features"]]
        print(f"downloaded {len(rows):,} / {len(ids):,}", end="\r")
    out = pd.DataFrame(rows)
    out["TRANSACTION_DATE"] = pd.to_datetime(out.TRANSACTION_DATE, unit="ms").dt.strftime("%Y-%m-%d")
    DATA.parent.mkdir(exist_ok=True)
    out.to_csv(DATA, index=False)


if not DATA.exists():
    download()
raw = pd.read_csv(DATA, parse_dates=["TRANSACTION_DATE"])
df = raw.rename(columns={"OBJECTID": "txn_id", "AGENCY": "agency", "TRANSACTION_DATE": "date",
                         "TRANSACTION_AMOUNT": "amount", "VENDOR_NAME": "vendor",
                         "VENDOR_STATE_PROVINCE": "vendor_state", "MCC_DESCRIPTION": "mcc"})
df["agency"] = df.agency.str.replace(r"\s+", " ", regex=True).str.strip()
df["vendor"] = df.vendor.fillna("UNKNOWN").str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
df["mcc"] = df.mcc.fillna("UNKNOWN").str.replace(r"\s+", " ", regex=True).str.strip()
df["month"] = df.date.dt.strftime("%Y-%m")
df["weekday"] = df.date.dt.dayofweek
print(f"{len(df):,} transactions, {df.date.min().date()} to {df.date.max().date()}, "
      f"{(df.amount < 0).sum():,} refunds/credits")

# ---------------------------------------------------------------- 2. categorise
CATEGORIES = [  # first match wins; keywords searched (case-insensitive) in the merchant category description
    ("Air travel", r"AIRLINE|AIRWAYS|AIR LINES|\bAIR\b|JETBLUE|DELTA|UNITED|SOUTHWEST|AMERICAN AIRLINES|ALASKA|FRONTIER|SPIRIT"),
    ("Hotels & lodging", r"HOTEL|MOTEL|LODGING|RESORT|\bINNS?\b|MARRIOTT|HILTON|HYATT|SHERATON|WESTIN|CASINO|SUITES|EXTENDED STAY|DOUBLETREE|WYNDHAM|OPRYLAND|EMBASSY|HAMPTON|RADISSON|HOLIDAY"),
    ("Ground travel & freight", r"CAR RENTAL|RENT-A-CAR|TAXI|LIMOUSINE|RAILROAD|PASSENGER|BUS LINES|PARKING|TOLL|TRANSPORTATION|TRANSPRTN|TRAVEL AGENC|FREIGHT|TRUCKING|COURIER"),
    ("Food & dining", r"RESTAURANT|EATING|FAST FOOD|CATERER|BAKER|GROCERY|SUPERMARKET|FOOD|DAIRY|CANDY|DRINKING|BAR LOUNGE"),
    ("Books & education", r"BOOK|SCHOOL|EDUCATION|COLLEGE|TRAINING|PERIODICAL|NEWS|PUBLISHING"),
    ("Office supplies & printing", r"STATIONERY|OFFICE|PRINTING|COPY|BLUEPRINT|PHOTOGRAPHIC|PAPER|COMMERCIAL ART"),
    ("Software & IT", r"COMPUTER|SOFTWARE|DATA PROCESSING|PRCSNG|PROGRAMING|INFORMATION RETRIEVAL|ELECTRONIC|DIGITAL|APPLICATIONS"),
    ("Telecom & utilities", r"TELECOM|TELEPHONE|CABLE|SATELLITE|UTLTS|UTILIT|POSTAGE|POSTAL|DELIVERY"),
    ("Events & memberships", r"ORGANIZATION|MEMBERSHIP|ASSOCIATION|CLUB|THEATRICAL|TICKET|AMUSEMENT|RECREATION|SPORT|CHARITABLE|BAND|ENTRTN|ENTERTAIN|MOTION PICTURE|TOURIST|MEDIA"),
    ("Medical & lab supplies", r"MEDICAL|DENTAL|HOSP|DRUG|PHARMAC|OPHTHALMIC|LABORATOR|HEALTH|DOCTOR|NURS|VETERIN|CHEMICAL"),
    ("Professional services", r"PROFESSIONAL|BUSINESS SERVICE|CONSULT|LEGAL|ACCOUNTING|ADVERTIS|GOVERNMENT SERVICES|EMPLOYMENT|DETECT|SECURITY|REAL ESTATE|COUNSEL|OTHER SERVICES|DRY CLEAN|ARCHITECT|ENGINEERING|EXTERMINAT|LANDSCAP"),
    ("Equipment & facilities", r"EQUIPMENT|HARDWARE|PLUMBING|HEATING|ELECTRICAL|LUMBER|BUILDING|CONTRACTOR|HOME SUPPLY|TOOL|AUTO|PARTS|MACHINERY|INDUSTRIAL|FURNITURE|JANITOR|CLEANING|CONSTRUCTION|DURABLE|WAREHOUS"),
    ("Retail & general merchandise", r"STORE|SHOP|MERCHANDISE|RETAIL|WHOLESALE|DEPARTMENT|DISCOUNT|MARKET|BOUTIQUE|APPAREL|CLOTH|UNIFORM"),
]
mcc_cat = {}
for m in df.mcc.unique():
    mcc_cat[m] = next((c for c, rx in CATEGORIES if re.search(rx, m, flags=re.I)), "Other")
df["category"] = df.mcc.map(mcc_cat)
print(df[df.amount > 0].groupby("category").amount.agg(["count", "sum"]).sort_values("sum", ascending=False).round(0))

# ---------------------------------------------------------------- 2b. categorise from the vendor name (ML)
# Real expense tools often see only the merchant text. Train a text model on vendor names and test
# it on vendors it has never seen (grouped split), so the score reflects new merchants.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
vend = (df[(df.amount > 0) & (df.category != "Other")]
        .groupby("vendor").category.agg(lambda s: s.value_counts().index[0]).reset_index())
vend["clean"] = vend.vendor.str.replace(r"^(SQ|TST|IN|PP|SP|PAYPAL|STAX|AMZN|DD|WPY|CKO) ?\*", "", regex=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr, te = next(gss.split(vend, groups=vend.vendor))
txt_model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=3000, C=4))
txt_model.fit(vend.clean.iloc[tr], vend.category.iloc[tr])
proba = txt_model.predict_proba(vend.clean.iloc[te])
pred = txt_model.classes_[proba.argmax(1)]
conf = proba.max(1)
y_true = vend.category.iloc[te].values
hi = conf >= 0.6
text_model = {"vendors_train": int(len(tr)), "vendors_test": int(len(te)),
              "accuracy": round(100 * accuracy_score(y_true, pred), 1),
              "macro_f1": round(f1_score(y_true, pred, average="macro"), 3),
              "majority_baseline": round(100 * (y_true == vend.category.iloc[tr].value_counts().index[0]).mean(), 1),
              "high_conf_share": round(100 * hi.mean(), 1),
              "high_conf_accuracy": round(100 * (pred[hi] == y_true[hi]).mean(), 1)}
per_cat = (pd.DataFrame({"y": y_true, "p": pred}).assign(ok=lambda d: d.y == d.p)
           .groupby("y").ok.agg(["mean", "size"]).sort_values("size", ascending=False))
text_model["per_category"] = [{"category": c, "accuracy": round(100 * r["mean"], 1), "vendors": int(r["size"])}
                              for c, r in per_cat.iterrows()]
print("Vendor-name model:", {k: v for k, v in text_model.items() if k != "per_category"})
# fill the gaps: suggest a category for "Other" merchants where the model is confident
other_v = df[(df.category == "Other") & (df.amount > 0)].vendor.drop_duplicates()
op = txt_model.predict_proba(other_v.str.replace(r"^(SQ|TST|IN|PP|SP|PAYPAL|STAX|AMZN|DD|WPY|CKO) ?\*", "", regex=True))
sugg = pd.DataFrame({"vendor": other_v.values, "suggested": txt_model.classes_[op.argmax(1)], "confidence": op.max(1)})
sugg = sugg[sugg.confidence >= 0.6].sort_values("confidence", ascending=False)
text_model["other_suggested"] = int(len(sugg))
text_model["other_vendors"] = int(len(other_v))
text_model["examples"] = (sugg.drop_duplicates("suggested").head(8)
                          .assign(confidence=lambda d: (100 * d.confidence).round(0)).to_dict(orient="records"))

# ---------------------------------------------------------------- 3. policy rules
p = df.amount > 0
SENSITIVE = r"CASINO|BAR LOUNGE|NIGHTCLUB|TAVERN|PACKAGE STORES|LIQUOR|\bSPAS?\b|BEAUTY|BARBER|GIFT|JEWEL|CRUISE|AMUSEMENT|GOLF|DATING|BETTING"
df["r_over_limit"] = (p & (df.amount > GOODS_LIMIT)).astype(int)
df["r_near_limit"] = (p & (((df.amount > 0.95 * GOODS_LIMIT) & (df.amount <= GOODS_LIMIT)) |
                           ((df.amount > 0.95 * SERVICES_LIMIT) & (df.amount <= SERVICES_LIMIT)))).astype(int)
grp = df[p].groupby(["agency", "vendor", "date"]).amount
day_n, day_sum, day_max = grp.transform("count"), grp.transform("sum"), grp.transform("max")
df["r_split"] = 0
df.loc[day_n.index, "r_split"] = ((day_n >= 2) & (day_max <= GOODS_LIMIT) & (day_sum > GOODS_LIMIT)).astype(int)
dup = df[p].duplicated(["agency", "vendor", "date", "amount"], keep="first")
df["r_duplicate"] = 0
df.loc[dup.index, "r_duplicate"] = dup.astype(int)
df["r_sensitive"] = (p & df.mcc.str.contains(SENSITIVE, regex=True, case=False)).astype(int)
df["r_weekend"] = (p & (df.weekday >= 5)).astype(int)
df["r_round"] = (p & (df.amount >= 1000) & (df.amount % 100 == 0)).astype(int)
RULES = ["r_over_limit", "r_near_limit", "r_split", "r_duplicate", "r_sensitive", "r_weekend", "r_round"]
SERIOUS = ["r_over_limit", "r_near_limit", "r_split", "r_duplicate", "r_sensitive"]  # weekend / round alone are not breaches
df["any_flag"] = df[SERIOUS].max(axis=1)

con = sqlite3.connect(":memory:")
df.assign(date=df.date.dt.strftime("%Y-%m-%d")).to_sql("txns", con, index=False)
queries = [q.strip() for q in (HERE / "queries.sql").read_text().split(";") if "SELECT" in q]
sql = {}
for q in queries:
    title = next(l for l in q.splitlines() if l.startswith("-- Q")).lstrip("- ").strip()
    res = pd.read_sql(q, con)
    sql[title.split(".")[0]] = res
    print(f"\n{title}\n{res.head(15).to_string(index=False)}")

# ---------------------------------------------------------------- 4. anomaly detection
pur = df[p].copy()
pur["log_amount"] = np.log1p(pur.amount)
mcc_med = pur.groupby("mcc").log_amount.transform("median")
mcc_mad = pur.groupby("mcc").log_amount.transform(lambda s: (s - s.median()).abs().median()).replace(0, np.nan)
pur["amount_vs_mcc"] = ((pur.log_amount - mcc_med) / mcc_mad).fillna(0).clip(-10, 10)
pur["vendor_rarity"] = -np.log(pur.groupby(["agency", "vendor"]).amount.transform("count"))
pur["mcc_rarity"] = -np.log(pur.groupby(["agency", "mcc"]).amount.transform("count"))
pur["agency_amount_z"] = pur.groupby("agency").log_amount.transform(lambda s: (s - s.mean()) / s.std()).fillna(0)
pur["out_of_state"] = (~pur.vendor_state.isin(["DC", "MD", "VA"])).astype(int)
pur["day_of_month"] = pur.date.dt.day
FEATS = ["log_amount", "amount_vs_mcc", "vendor_rarity", "mcc_rarity", "agency_amount_z", "out_of_state", "day_of_month"]

iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
iso.fit(pur[FEATS])
pur["iso_score"] = -iso.score_samples(pur[FEATS])
pur["iso_pct"] = pur.iso_score.rank(pct=True)
pur["zscore_baseline"] = pur.amount_vs_mcc.abs()

# weak labels: rule-confirmed problems that do not use the model's features directly
weak = ((pur.r_split + pur.r_duplicate + pur.r_over_limit + pur.r_sensitive) > 0).astype(int)
def precision_at(score, k):
    top = score.sort_values(ascending=False).index[:k]
    return round(100 * weak.loc[top].mean(), 1)
eval_rows = []
for name, score in [("Isolation Forest", pur.iso_score), ("Z-score baseline (amount vs category)", pur.zscore_baseline),
                    ("Random order", pd.Series(np.random.default_rng(42).random(len(pur)), index=pur.index))]:
    eval_rows.append({"method": name, "pr_auc": round(average_precision_score(weak, score), 3),
                      "precision_at_500": precision_at(score, 500), "precision_at_2000": precision_at(score, 2000)})
    print(eval_rows[-1])
print(f"Weak-label base rate: {100 * weak.mean():.1f}%")

# ---------------------------------------------------------------- 5. Benford's law
BENF = np.log10(1 + 1 / np.arange(1, 10))
def first_digit(a):
    s = a[a >= 10].astype(str).str.lstrip("0.").str[0]
    return s[s.str.isdigit()].astype(int)
def benford_mad(a):
    d = first_digit(a)
    obs = d.value_counts(normalize=True).reindex(range(1, 10), fill_value=0).values
    return float(np.mean(np.abs(obs - BENF))), obs
overall_mad, overall_obs = benford_mad(pur.amount)
benford_ag = []
for ag, g in pur.groupby("agency"):
    if len(g) >= 1000:
        mad, _ = benford_mad(g.amount)
        benford_ag.append({"agency": ag, "purchases": len(g), "mad": round(mad, 4),
                           "verdict": "Close" if mad < 0.006 else "Acceptable" if mad < 0.012 else
                           "Marginal" if mad < 0.015 else "Nonconformity"})
benford_ag.sort(key=lambda r: -r["mad"])
print(f"\nBenford overall MAD {overall_mad:.4f}", benford_ag[:5])

# ---------------------------------------------------------------- 6. risk score + review queue
POINTS = {"r_over_limit": 3, "r_split": 3, "r_duplicate": 2, "r_near_limit": 2, "r_sensitive": 2, "r_weekend": 1, "r_round": 1}
REASON = {"r_over_limit": "over the $5,000 single-purchase limit",
          "r_split": "one of several same-day charges to this vendor that together exceed $5,000",
          "r_duplicate": "same vendor, day and exact amount as another charge",
          "r_near_limit": "just under a purchase limit",
          "r_sensitive": "sensitive merchant category",
          "r_weekend": "weekend purchase", "r_round": "round amount"}
pur["rule_points"] = sum(pur[r] * w for r, w in POINTS.items())
pur["risk_score"] = (pur.rule_points + 4 * pur.iso_pct).round(2)

def reasons(r):
    out = [REASON[k] for k in POINTS if r[k]]
    if r.iso_pct >= 0.99:
        out.append(f"unusual for this agency (anomaly score top {max(1, round(100 * (1 - r.iso_pct)))}%)")
    return "; ".join(out).capitalize()
queue = pur.sort_values("risk_score", ascending=False).head(25).copy()
queue["reasons"] = queue.apply(reasons, axis=1)
review_queue = [{"date": r.date.strftime("%Y-%m-%d"), "agency": r.agency, "vendor": r.vendor, "category": r.category,
                 "amount": round(r.amount, 2), "score": r.risk_score, "reasons": r.reasons} for r in queue.itertuples()]
top_pct_cut = pur.risk_score.quantile(0.99)
reviewed = pur[pur.risk_score >= top_pct_cut]
print(f"\nTop 1% review queue: {len(reviewed):,} purchases, ${reviewed.amount.sum():,.0f}")
print(queue[["date", "agency", "vendor", "amount", "risk_score", "reasons"]].head(10).to_string())

score_bins = pd.cut(pur.risk_score, bins=[-0.01, 2, 3, 4, 5, 6, 20], labels=["0–2", "2–3", "3–4", "4–5", "5–6", "6+"])
score_dist = pur.groupby(score_bins, observed=False).agg(purchases=("amount", "size"), spend=("amount", "sum")).reset_index()

# ---------------------------------------------------------------- 7. export
q1 = sql["Q1"].iloc[0]
q3 = sql["Q3"].set_index("band").purchases
q4 = sql["Q4"].iloc[0]
q5 = sql["Q5"].iloc[0]
results = {
    "kpis": {"transactions": int(q1.transactions), "purchases": int(q1.purchases), "spend": float(q1.spend),
             "agencies": int(q1.agencies), "vendors": int(q1.vendors),
             "period": f"{df.date.min():%b %Y} – {df.date.max():%b %Y}",
             "flagged_pct": round(100 * pur.any_flag.mean(), 1),
             "bunching_ratio": round(q3.get("$4,750–5,000", 0) / max(1, q3.get("$5,000–5,250", 1)), 1),
             "split_groups": int(q4.split_groups), "split_spend": float(q4.spend_involved),
             "dup_extra": int(q5.extra_charges), "dup_spend": float(q5.extra_spend),
             "review_n": int(len(reviewed)), "review_spend": round(float(reviewed.amount.sum())),
             "weak_base": round(100 * weak.mean(), 1), "benford_mad": round(overall_mad, 4),
             "categories": int(df.category.nunique()), "mcc_count": int(df.mcc.nunique()),
             "other_share": round(100 * (pur.category == "Other").mean(), 1)},
    "sql": {k: v.replace({np.nan: None}).to_dict(orient="records") for k, v in sql.items()},
    "anomaly_eval": eval_rows, "text_model": text_model, "benford": {"expected": BENF.round(4).tolist(), "observed": overall_obs.round(4).tolist(),
                                            "agencies": benford_ag},
    "review_queue": review_queue,
    "score_dist": [{"band": str(r.risk_score), "purchases": int(r.purchases), "spend": round(float(r.spend))}
                   for r in score_dist.itertuples()],
}
(HERE / "results.json").write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
pur[["txn_id", "date", "agency", "vendor", "category", "amount", "risk_score"] + RULES].to_csv(HERE / "scored_purchases.csv", index=False)
print("\nWrote results.json")
