-- Corporate Card Expense Copilot: SQL policy engine (SQLite)
-- Table: txns (one row per DC government purchase-card transaction, Jun 2023 - Sep 2026)
-- Card rules used (DC Public Schools purchase-card guidance): single purchase up to $5,000 for goods,
-- $2,500 for services. The data has no cardholder ID, so rules work at agency level.

-- Q1. Programme overview
SELECT COUNT(*)                                         AS transactions,
       SUM(amount > 0)                                  AS purchases,
       SUM(amount < 0)                                  AS refunds,
       ROUND(SUM(CASE WHEN amount > 0 THEN amount END)) AS spend,
       ROUND(AVG(CASE WHEN amount > 0 THEN amount END), 2) AS avg_purchase,
       COUNT(DISTINCT agency)                           AS agencies,
       COUNT(DISTINCT vendor)                           AS vendors
FROM txns;

-- Q2. Spend by expense category (auto-categorised from the merchant category code)
SELECT category,
       COUNT(*)                                            AS purchases,
       ROUND(SUM(amount))                                  AS spend,
       ROUND(100.0 * SUM(amount) / SUM(SUM(amount)) OVER (), 1) AS spend_share_pct,
       ROUND(AVG(amount), 0)                               AS avg_purchase
FROM txns
WHERE amount > 0
GROUP BY category
ORDER BY spend DESC;

-- Q3. Bunching just under the limits: purchases in $250 bands around $2,500 and $5,000
SELECT CASE WHEN amount > 2250 AND amount <= 2500 THEN '$2,250–2,500'
            WHEN amount > 2500 AND amount <= 2750 THEN '$2,500–2,750'
            WHEN amount > 4750 AND amount <= 5000 THEN '$4,750–5,000'
            WHEN amount > 5000 AND amount <= 5250 THEN '$5,000–5,250' END AS band,
       COUNT(*) AS purchases
FROM txns
WHERE amount > 2250 AND amount <= 5250 AND (amount <= 2750 OR amount > 4750)
GROUP BY band
ORDER BY band;

-- Q4. Rule: possible split purchases (same agency + vendor + day, 2+ charges each within $5,000, total above $5,000)
WITH day_vendor AS (
  SELECT agency, vendor, date, COUNT(*) AS charges, SUM(amount) AS total, MAX(amount) AS largest
  FROM txns
  WHERE amount > 0
  GROUP BY agency, vendor, date
)
SELECT COUNT(*)            AS split_groups,
       SUM(charges)        AS charges_involved,
       ROUND(SUM(total))   AS spend_involved,
       ROUND(AVG(charges), 1) AS avg_charges_per_group
FROM day_vendor
WHERE charges >= 2 AND largest <= 5000 AND total > 5000;

-- Q5. Rule: possible duplicate charges (same agency, vendor, day and exact amount)
SELECT COUNT(*) AS duplicate_groups, SUM(n - 1) AS extra_charges, ROUND(SUM((n - 1) * amount)) AS extra_spend
FROM (SELECT agency, vendor, date, amount, COUNT(*) AS n
      FROM txns WHERE amount > 0
      GROUP BY agency, vendor, date, amount HAVING n >= 2);

-- Q6. Rule hits summary (flags computed in Python from the same logic, stored as 0/1 columns)
SELECT 'Over $5,000 single-purchase limit' AS rule, SUM(r_over_limit) AS hits, ROUND(SUM(r_over_limit * amount)) AS spend FROM txns
UNION ALL SELECT 'Just under a limit ($4,750–5,000 or $2,375–2,500)', SUM(r_near_limit), ROUND(SUM(r_near_limit * amount)) FROM txns
UNION ALL SELECT 'Possible split purchase', SUM(r_split), ROUND(SUM(r_split * amount)) FROM txns
UNION ALL SELECT 'Possible duplicate charge', SUM(r_duplicate), ROUND(SUM(r_duplicate * amount)) FROM txns
UNION ALL SELECT 'Sensitive merchant (casino hotel, bar, spa, gift shop…)', SUM(r_sensitive), ROUND(SUM(r_sensitive * amount)) FROM txns
UNION ALL SELECT 'Weekend purchase', SUM(r_weekend), ROUND(SUM(r_weekend * amount)) FROM txns
UNION ALL SELECT 'Round amount of $1,000+', SUM(r_round), ROUND(SUM(r_round * amount)) FROM txns;

-- Q7. Agency scorecard: spend, share of purchases flagged, near-limit share, ranked (RANK)
SELECT agency,
       COUNT(*)                                   AS purchases,
       ROUND(SUM(amount))                         AS spend,
       ROUND(100.0 * AVG(any_flag), 1)            AS flagged_pct,
       ROUND(100.0 * AVG(r_near_limit), 2)        AS near_limit_pct,
       SUM(r_split)                               AS split_charges,
       RANK() OVER (ORDER BY AVG(any_flag) DESC)  AS risk_rank
FROM txns
WHERE amount > 0
GROUP BY agency
HAVING purchases >= 1000
ORDER BY flagged_pct DESC;

-- Q8. Monthly spend and flagged spend, with a 3-month moving average of the flagged share
SELECT month,
       ROUND(SUM(amount))                        AS spend,
       ROUND(SUM(any_flag * amount))             AS flagged_spend,
       ROUND(100.0 * SUM(any_flag * amount) / SUM(amount), 1) AS flagged_share_pct,
       ROUND(AVG(100.0 * SUM(any_flag * amount) / SUM(amount)) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 1) AS flagged_share_3m_avg
FROM txns
WHERE amount > 0
GROUP BY month
ORDER BY month;

-- Q9. Top vendors by spend, with how concentrated spend is (cumulative share, window SUM)
SELECT vendor, purchases, spend,
       ROUND(100.0 * SUM(spend) OVER (ORDER BY spend DESC ROWS UNBOUNDED PRECEDING) / SUM(spend) OVER (), 1) AS cumulative_share_pct
FROM (SELECT vendor, COUNT(*) AS purchases, ROUND(SUM(amount)) AS spend FROM txns WHERE amount > 0 GROUP BY vendor)
ORDER BY spend DESC
LIMIT 10;
