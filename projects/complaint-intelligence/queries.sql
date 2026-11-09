-- Credit Card Complaint Intelligence: SQL analysis (SQLite)
-- Table: complaints (one row per CFPB complaint about a credit card, Aug 2023 - Oct 2026)
-- issuer: Amex, Chase, Capital One, Citi
-- relief = 1 if the company closed the complaint with monetary or non-monetary relief

-- Q1. Issuer overview: volume, relief rate, timeliness
SELECT issuer,
       COUNT(*)                                             AS complaints,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)   AS share_pct,
       ROUND(100.0 * AVG(relief), 1)                        AS relief_pct,
       ROUND(100.0 * AVG(monetary), 1)                      AS monetary_relief_pct,
       ROUND(100.0 * AVG(timely), 2)                        AS timely_pct
FROM complaints
GROUP BY issuer
ORDER BY complaints DESC;

-- Q2. Issue mix: what share of each issuer's complaints is about each issue
SELECT issuer, issue_short,
       COUNT(*)                                                                 AS complaints,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY issuer), 1)    AS share_pct
FROM complaints
GROUP BY issuer, issue_short
ORDER BY issuer, complaints DESC;

-- Q3. Where Amex over-indexes: Amex issue share vs the average of the 3 peers (index 100 = same)
WITH counts AS (
  SELECT CASE WHEN issuer = 'Amex' THEN 1 ELSE 0 END AS is_amex, issue_short, COUNT(*) AS n
  FROM complaints
  GROUP BY is_amex, issue_short
),
mix AS (
  SELECT is_amex, issue_short, 1.0 * n / SUM(n) OVER (PARTITION BY is_amex) AS share
  FROM counts
)
SELECT a.issue_short,
       ROUND(100 * a.share, 1)                 AS amex_share_pct,
       ROUND(100 * p.share, 1)                 AS peer_share_pct,
       ROUND(100 * a.share / p.share, 0)       AS amex_index
FROM mix a JOIN mix p ON a.issue_short = p.issue_short AND a.is_amex = 1 AND p.is_amex = 0
WHERE a.share >= 0.01
ORDER BY amex_index DESC;

-- Q4. Monthly complaint volume per issuer, with month-on-month change (LAG)
SELECT issuer, month, complaints,
       complaints - LAG(complaints) OVER (PARTITION BY issuer ORDER BY month) AS mom_change
FROM (SELECT issuer, month, COUNT(*) AS complaints FROM complaints GROUP BY issuer, month)
ORDER BY issuer, month;

-- Q5. Amex: year-on-year growth by issue (last 12 months vs the 12 months before)
SELECT issue_short,
       SUM(month BETWEEN '2025-11' AND '2026-10')                  AS last_12m,
       SUM(month BETWEEN '2024-11' AND '2025-10')                  AS prior_12m,
       ROUND(100.0 * (SUM(month BETWEEN '2025-11' AND '2026-10')
             - SUM(month BETWEEN '2024-11' AND '2025-10'))
             / NULLIF(SUM(month BETWEEN '2024-11' AND '2025-10'), 0), 0) AS yoy_pct
FROM complaints
WHERE issuer = 'Amex'
GROUP BY issue_short
HAVING prior_12m >= 50
ORDER BY yoy_pct DESC;

-- Q6. Amex root causes: top sub-issues, how often they end in relief, and their rank
SELECT sub_issue,
       COUNT(*)                                        AS complaints,
       ROUND(100.0 * AVG(relief), 1)                   AS relief_pct,
       RANK() OVER (ORDER BY COUNT(*) DESC)            AS volume_rank
FROM complaints
WHERE issuer = 'Amex' AND sub_issue IS NOT NULL
GROUP BY sub_issue
ORDER BY complaints DESC
LIMIT 12;

-- Q7. Relief rate by issue for each issuer (which complaint types most often end in a fix or refund?)
SELECT issue_short,
       ROUND(100.0 * AVG(CASE WHEN issuer = 'Amex'        THEN relief END), 1) AS amex_relief_pct,
       ROUND(100.0 * AVG(CASE WHEN issuer = 'Chase'       THEN relief END), 1) AS chase_relief_pct,
       ROUND(100.0 * AVG(CASE WHEN issuer = 'Capital One' THEN relief END), 1) AS capital_one_relief_pct,
       ROUND(100.0 * AVG(CASE WHEN issuer = 'Citi'        THEN relief END), 1) AS citi_relief_pct,
       SUM(issuer = 'Amex')                                                    AS amex_complaints
FROM complaints
GROUP BY issue_short
HAVING amex_complaints >= 100
ORDER BY amex_complaints DESC;

-- Q7b. Late responses (company missed the 15-day CFPB deadline) by issuer
SELECT issuer, SUM(1 - timely) AS late_responses, COUNT(*) AS complaints,
       ROUND(100.0 * SUM(1 - timely) / SUM(SUM(1 - timely)) OVER (), 1) AS share_of_all_late_pct
FROM complaints
GROUP BY issuer
ORDER BY late_responses DESC;

-- Q7c. When did Amex's late responses happen? (months with at least one late response)
SELECT month, SUM(1 - timely) AS late_responses, COUNT(*) AS complaints,
       SUM(issue_short = 'Disputed purchase') AS disputed_purchase_complaints
FROM complaints
WHERE issuer = 'Amex'
GROUP BY month
HAVING late_responses > 0
ORDER BY month;

-- Q8. Vulnerable customers: Older Americans and servicemembers (Amex)
SELECT CASE WHEN tags IS NULL THEN 'No tag'
            WHEN tags LIKE '%Older American%' AND tags LIKE '%Servicemember%' THEN 'Older American + Servicemember'
            WHEN tags LIKE '%Older American%' THEN 'Older American'
            ELSE 'Servicemember' END                   AS customer_group,
       COUNT(*)                                        AS complaints,
       ROUND(100.0 * AVG(relief), 1)                   AS relief_pct
FROM complaints
WHERE issuer = 'Amex'
GROUP BY customer_group
ORDER BY complaints DESC;

-- Q9. Amex complaints by state: top 10, and a 3-month rolling average for the top state
SELECT state, COUNT(*) AS complaints,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS share_pct
FROM complaints
WHERE issuer = 'Amex' AND state IS NOT NULL
GROUP BY state
ORDER BY complaints DESC
LIMIT 10;

-- Q10. Amex monthly volume with a 3-month moving average (window frame)
SELECT month, COUNT(*) AS complaints,
       ROUND(AVG(COUNT(*)) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 0) AS moving_avg_3m
FROM complaints
WHERE issuer = 'Amex'
GROUP BY month
ORDER BY month;
