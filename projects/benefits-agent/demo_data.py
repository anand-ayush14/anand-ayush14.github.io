"""
Generate SYNTHETIC card transactions for 3 demo members, Jan 1 – Nov 9, 2026.
These are made-up people and purchases, used only to demonstrate the agent. No real member data.
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
rng = np.random.default_rng(7)
END = date(2026, 11, 9)
rows = []


def add(member, card, d, merchant, amount):
    if d <= END:
        rows.append({"member": member, "card": card, "date": d.isoformat(), "merchant": merchant, "amount": round(amount, 2)})


def months():
    return [date(2026, m, 1) for m in range(1, END.month + 1)]


def day(m, lo=1, hi=27):
    return m + timedelta(days=int(rng.integers(lo - 1, hi)))


# ---- Member A: "Priya", Platinum + Gold, eats out a lot, travels twice a year
for m in months():
    add("A", "platinum", day(m, 1, 5), "DISNEY+", 13.99)                      # uses only half the $25 digital credit
    if rng.random() < 0.8: add("A", "platinum", day(m), "UBER TRIP", rng.uniform(12, 40))
    if rng.random() < 0.5: add("A", "gold", day(m), "UBER EATS", rng.uniform(15, 35))
    add("A", "gold", day(m), "RESY - LOCAL BISTRO", rng.uniform(60, 140))        # Resy on Gold, not Platinum
    if rng.random() < 0.6: add("A", "gold", day(m), "GRUBHUB", rng.uniform(18, 30))
    if rng.random() < 0.3: add("A", "gold", day(m), "DUNKIN #3321", rng.uniform(4, 9))
    add("A", "gold", day(m), "WHOLE FOODS", rng.uniform(80, 220))
    if m.month in (2, 5, 8): add("A", "platinum", day(m), "RESY - OMAKASE BAR", rng.uniform(150, 260))
add("A", "platinum", date(2026, 1, 12), "WALMART+", 12.95)  # signed up, then cancelled
add("A", "platinum", date(2026, 3, 18), "AMEX TRAVEL FHR - HOTEL", 820)
add("A", "platinum", date(2026, 4, 2), "DELTA AIR LINES", 640)
add("A", "platinum", date(2026, 4, 2), "DELTA FEE - SEAT UPGRADE", 89)
add("A", "platinum", date(2026, 6, 20), "LULULEMON", 128)
add("A", "platinum", date(2026, 1, 20), "UBER ONE", 9.99)
for m in months()[1:]:
    add("A", "platinum", m + timedelta(days=19), "UBER ONE", 9.99)

# ---- Member B: "Rahul", Gold only, light user
for m in months():
    if rng.random() < 0.7: add("B", "gold", day(m), "UBER TRIP", rng.uniform(8, 25))
    if rng.random() < 0.25: add("B", "gold", day(m), "FIVE GUYS", rng.uniform(12, 22))
    add("B", "gold", day(m), "KROGER", rng.uniform(60, 160))
    add("B", "gold", day(m), "LOCAL RESTAURANT", rng.uniform(25, 60))
add("B", "gold", date(2026, 9, 14), "RESY - PIZZERIA", 72)

# ---- Member C: "Sam", Delta Reserve + Platinum, road warrior
for m in months():
    for _ in range(int(rng.integers(3, 7))):
        add("C", "delta_reserve", day(m), rng.choice(["UBER TRIP", "LYFT RIDE"]), rng.uniform(15, 55))
    if rng.random() < 0.35: add("C", "delta_reserve", day(m), "RESY - STEAKHOUSE", rng.uniform(90, 200))
    add("C", "delta_reserve", day(m), "DELTA AIR LINES", rng.uniform(250, 900))
    add("C", "platinum", day(m, 1, 3), "YOUTUBE TV", 82.99)
    add("C", "platinum", day(m, 3, 6), "NYTIMES", 6.25)
    if rng.random() < 0.5: add("C", "platinum", day(m), "UBER EATS", rng.uniform(15, 45))
add("C", "platinum", date(2026, 1, 8), "CLEAR", 209)
add("C", "platinum", date(2026, 2, 3), "DELTA FEE - CHECKED BAG", 35)
add("C", "platinum", date(2026, 5, 11), "DELTA FEE - SKY CLUB DAY PASS", 50)
add("C", "platinum", date(2026, 8, 22), "DELTA FEE - SEAT UPGRADE", 129)
add("C", "platinum", date(2026, 2, 16), "AMEX TRAVEL FHR - HOTEL", 1150)
add("C", "platinum", date(2026, 9, 5), "AMEX TRAVEL THC - HOTEL", 690)
add("C", "delta_reserve", date(2026, 7, 9), "DELTA STAYS - HOTEL", 310)
add("C", "platinum", date(2026, 3, 30), "OURA", 349)

out = HERE / "data"
out.mkdir(exist_ok=True)
pd.DataFrame(rows).sort_values(["member", "date"]).to_csv(out / "transactions.csv", index=False)
MEMBERS = {"A": {"name": "Priya (demo)", "renewals": {"platinum": "2026-12", "gold": "2027-03"}},
           "B": {"name": "Rahul (demo)", "renewals": {"gold": "2027-01"}},
           "C": {"name": "Sam (demo)", "renewals": {"delta_reserve": "2027-02", "platinum": "2026-12"}}}
pd.Series(MEMBERS).to_json(out / "members.json", indent=1)
print(f"{len(rows)} synthetic transactions written")
