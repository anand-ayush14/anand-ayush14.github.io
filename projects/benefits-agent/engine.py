"""
Benefits engine: the tools the agent calls.

Given public card benefit rules (cards.json) and a member's card transactions, work out
which statement credits have been used, what is left in the current period, what expires
soon, which card to use for a purchase, and whether each card is worth its annual fee.
All functions return plain dicts/lists so they can be exposed as MCP or LLM tools.
"""
import calendar
import json
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CARDS = {c["id"]: c for c in json.loads((HERE / "cards.json").read_text())["cards"]}
POINT_VALUE = {"platinum": 0.01, "gold": 0.01, "delta_reserve": 0.011}  # conservative $ per point/mile


def load_transactions(path=HERE / "data" / "transactions.csv"):
    t = pd.read_csv(path, parse_dates=["date"])
    t["merchant"] = t.merchant.str.upper()
    return t


def period_of(period, d):
    """Return (start, end, label) of the benefit period that contains date d (calendar periods)."""
    y, m = d.year, d.month
    if period == "monthly":
        s, e, label = date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1]), d.strftime("%b %Y")
    elif period == "quarterly":
        q = (m - 1) // 3
        s, e, label = date(y, 3 * q + 1, 1), date(y, 3 * q + 3, calendar.monthrange(y, 3 * q + 3)[1]), f"Q{q + 1} {y}"
    elif period == "semiannual":
        h = 0 if m <= 6 else 1
        s, e, label = (date(y, 1, 1), date(y, 6, 30), f"Jan–Jun {y}") if h == 0 else (date(y, 7, 1), date(y, 12, 31), f"Jul–Dec {y}")
    else:
        s, e, label = date(y, 1, 1), date(y, 12, 31), str(y)
    return s, e, label


def periods_in_year(period, year, upto):
    """All benefit periods of a calendar year that have started by `upto`."""
    starts = {"monthly": range(1, 13), "quarterly": (1, 4, 7, 10), "semiannual": (1, 7), "annual": (1,)}[period]
    return [period_of(period, date(year, m, 1)) for m in starts if date(year, m, 1) <= upto]


def credit_amount(credit, start):
    if credit.get("december_amount") and start.month == 12:
        return credit["december_amount"]
    return credit["amount"]


def _matches(merchant, keywords):
    return max((len(k) for k in keywords if k in merchant), default=0)


def assign_credits(tx):
    """Tag each transaction with the single credit it can count towards on the card used (most specific match)."""
    out = []
    for r in tx.itertuples():
        card = CARDS.get(r.card)
        best, best_len = None, 0
        for c in card["credits"] if card else []:
            L = _matches(r.merchant, c["merchants"])
            if L > best_len:
                best, best_len = c["id"], L
        out.append(best)
    return tx.assign(credit_id=out)


def credit_usage(tx, member, as_of):
    """Every credit-period for a member this year: available, used, remaining, status."""
    t = assign_credits(tx[(tx.member == member) & (tx.date.dt.date <= as_of)])
    held = tx[tx.member == member].card.unique()
    rows = []
    for card_id in held:
        for c in CARDS[card_id]["credits"]:
            for s, e, label in periods_in_year(c["period"], as_of.year, as_of):
                amt = credit_amount(c, s)
                spent = t[(t.credit_id == c["id"]) & (t.date.dt.date >= s) & (t.date.dt.date <= e)].amount.sum()
                used = round(min(amt, spent), 2)
                status = "open" if e >= as_of else ("used" if used >= amt - 0.01 else "missed")
                rows.append({"card": card_id, "card_name": CARDS[card_id]["name"], "credit_id": c["id"], "credit": c["name"],
                             "period": label, "period_type": c["period"], "start": s.isoformat(), "end": e.isoformat(),
                             "available": amt, "used": used, "remaining": round(amt - used, 2),
                             "days_left": (e - as_of).days if e >= as_of else 0, "status": status, "enroll": c["enroll"]})
    return rows


# ------------------------------------------------------------------ tools
def _py(o):
    """Convert numpy numbers to plain Python so results serialise as JSON."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_py(v) for v in o]
    return o.item() if hasattr(o, "item") else o


def get_unused_credits(tx, member, as_of):
    """Credits still open in the current period with money left, soonest-expiring first."""
    rows = [r for r in credit_usage(tx, member, as_of) if r["status"] == "open" and r["remaining"] > 0]
    return sorted(rows, key=lambda r: (r["days_left"], -r["remaining"]))


def year_to_date(tx, member, as_of):
    """Per credit: total available so far this year, used, and missed (expired unused)."""
    df = pd.DataFrame(credit_usage(tx, member, as_of))
    g = df.groupby(["card", "card_name", "credit"]).agg(
        available=("available", "sum"), used=("used", "sum"),
        missed=("remaining", lambda s: s[df.loc[s.index, "status"] == "missed"].sum())).reset_index()
    g["use_rate"] = (100 * g.used / g.available).round(1)
    return _py(g.sort_values("missed", ascending=False).round(2).to_dict(orient="records"))


def value_vs_fee(tx, member, as_of):
    """Credits captured year to date, annualised, vs each card's annual fee."""
    ytd = pd.DataFrame(year_to_date(tx, member, as_of))
    months = as_of.month - 1 + as_of.day / calendar.monthrange(as_of.year, as_of.month)[1]
    out = []
    for card_id, g in ytd.groupby("card"):
        card = CARDS[card_id]
        annual_potential = sum(credit_amount(c, date(as_of.year, m, 1)) * {"monthly": 1, "quarterly": 1 / 3, "semiannual": 1 / 6, "annual": 1 / 12}[c["period"]]
                               for c in card["credits"] for m in range(1, 13))
        captured_rate = g.used.sum() / g.available.sum() if g.available.sum() else 0
        projected = round(captured_rate * annual_potential, 0)
        net = projected - card["annual_fee"]
        out.append({"card": card_id, "card_name": card["name"], "annual_fee": card["annual_fee"],
                    "credits_used_ytd": round(g.used.sum(), 2), "credits_missed_ytd": round(g.missed.sum(), 2),
                    "capture_rate_pct": round(100 * captured_rate, 1), "annual_credit_potential": round(annual_potential, 0),
                    "projected_annual_credit_value": projected, "net_vs_fee": net,
                    "verdict": "Credits alone cover the fee" if net >= 0 else
                               "Credits cover most of the fee; points and lounge access decide" if net > -0.4 * card["annual_fee"] else
                               "At risk: credits cover less than 60% of the fee"})
    return _py(out)


def recommend_card_for_purchase(tx, member, as_of, merchant, amount):
    """Which of the member's cards gives the most value for this purchase (open credit + points)."""
    merchant = merchant.upper()
    open_credits = {r["credit_id"]: r for r in get_unused_credits(tx, member, as_of)}
    options = []
    for card_id in tx[tx.member == member].card.unique():
        card = CARDS[card_id]
        credit_val, credit_name = 0.0, None
        best_len = 0
        for c in card["credits"]:
            L = _matches(merchant, c["merchants"])
            if L > best_len and c["id"] in open_credits:
                best_len, credit_val, credit_name = L, min(amount, open_credits[c["id"]]["remaining"]), c["name"]
        rate = max([e["rate"] for e in card["earn"] if _matches(merchant, e["match"])], default=1)
        points_val = round(amount * rate * POINT_VALUE[card_id], 2)
        options.append({"card": card_id, "card_name": card["name"], "credit_used": credit_name,
                        "credit_value": round(credit_val, 2), "points_rate": rate, "points_value": points_val,
                        "total_value": round(credit_val + points_val, 2)})
    options.sort(key=lambda o: -o["total_value"])
    return _py({"merchant": merchant, "amount": amount, "best": options[0] if options else None, "options": options})
