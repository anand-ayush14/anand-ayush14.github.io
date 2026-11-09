"""Quick checks of the benefits engine (run: python test_engine.py)."""
from datetime import date
import engine

tx = engine.load_transactions()
d = date(2026, 11, 9)
assert engine.period_of("quarterly", d)[2] == "Q4 2026"
assert engine.period_of("semiannual", date(2026, 3, 1))[1] == date(2026, 6, 30)
plat = engine.CARDS["platinum"]
uber = next(c for c in plat["credits"] if c["id"] == "plat_uber")
assert engine.credit_amount(uber, date(2026, 12, 1)) == 35 and engine.credit_amount(uber, date(2026, 5, 1)) == 15
# a $9.99 "UBER ONE" charge counts towards Uber One, not Uber Cash (most specific match wins)
t = engine.assign_credits(tx[(tx.member == "A") & (tx.merchant == "UBER ONE")])
assert set(t.credit_id) == {"plat_uberone"}
# Platinum annual credit potential matches the public total (~$2,684 incl. Walmart+ at $12.95/month)
fee = {f["card"]: f for f in engine.value_vs_fee(tx, "A", d)}
assert fee["platinum"]["annual_credit_potential"] == 2684 and fee["gold"]["annual_credit_potential"] == 424
# recommending a Resy dinner for member A picks Platinum (open $100 quarterly Resy credit)
assert engine.recommend_card_for_purchase(tx, "A", d, "RESY - TEST", 120)["best"]["card"] == "platinum"
print("all tests passed")
