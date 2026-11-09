-- Credit Card Customer Attrition: SQL analysis (SQLite)
-- Table: customers (one row per card holder, last 12 months of activity)
-- attrited = 1 if the customer closed their card account

-- Q1. Portfolio overview
SELECT COUNT(*)                                        AS customers,
       SUM(attrited)                                   AS attrited,
       ROUND(100.0 * AVG(attrited), 1)                 AS attrition_rate_pct,
       ROUND(AVG(credit_limit), 0)                     AS avg_credit_limit,
       ROUND(AVG(total_trans_amt), 0)                  AS avg_annual_spend,
       ROUND(AVG(total_trans_ct), 1)                   AS avg_annual_txns
FROM customers;

-- Q2. Attrition by number of transactions in the last 12 months
SELECT CASE WHEN total_trans_ct < 40  THEN '1. <40'
            WHEN total_trans_ct < 60  THEN '2. 40-59'
            WHEN total_trans_ct < 80  THEN '3. 60-79'
            WHEN total_trans_ct < 100 THEN '4. 80-99'
            ELSE                           '5. 100+' END AS txn_band,
       COUNT(*)                                        AS customers,
       ROUND(100.0 * AVG(attrited), 1)                 AS attrition_rate_pct
FROM customers
GROUP BY txn_band
ORDER BY txn_band;

-- Q3. Attrition by change in transaction count, Q4 vs Q1 (early-warning signal)
SELECT CASE WHEN ct_chng_q4_q1 < 0.5 THEN '1. fell >50%'
            WHEN ct_chng_q4_q1 < 0.7 THEN '2. fell 30-50%'
            WHEN ct_chng_q4_q1 < 0.9 THEN '3. fell 10-30%'
            ELSE                          '4. stable / up' END AS activity_trend,
       COUNT(*)                                        AS customers,
       ROUND(100.0 * AVG(attrited), 1)                 AS attrition_rate_pct
FROM customers
GROUP BY activity_trend
ORDER BY activity_trend;

-- Q4. Attrition by contacts with the bank in 12 months (complaints / service issues)
SELECT contacts_12m,
       COUNT(*)                                        AS customers,
       ROUND(100.0 * AVG(attrited), 1)                 AS attrition_rate_pct
FROM customers
GROUP BY contacts_12m
ORDER BY contacts_12m;

-- Q5. Attrition by number of products held with the bank
SELECT products_held,
       COUNT(*)                                        AS customers,
       ROUND(100.0 * AVG(attrited), 1)                 AS attrition_rate_pct
FROM customers
GROUP BY products_held
ORDER BY products_held;

-- Q6. Attrition by revolving balance (customers who carry no balance leave more)
SELECT CASE WHEN revolving_bal = 0    THEN '1. $0 (pays in full / unused)'
            WHEN revolving_bal < 1000 THEN '2. $1-999'
            WHEN revolving_bal < 2000 THEN '3. $1,000-1,999'
            ELSE                           '4. $2,000+' END AS revolving_band,
       COUNT(*)                                        AS customers,
       ROUND(100.0 * AVG(attrited), 1)                 AS attrition_rate_pct
FROM customers
GROUP BY revolving_band
ORDER BY revolving_band;

-- Q7. Attrition by months inactive in the last 12 months
SELECT months_inactive,
       COUNT(*)                                        AS customers,
       ROUND(100.0 * AVG(attrited), 1)                 AS attrition_rate_pct
FROM customers
GROUP BY months_inactive
ORDER BY months_inactive;

-- Q8. Card tier: attrition and share of all lost customers (window function)
SELECT card_category,
       COUNT(*)                                        AS customers,
       SUM(attrited)                                   AS attrited,
       ROUND(100.0 * AVG(attrited), 1)                 AS attrition_rate_pct,
       ROUND(100.0 * SUM(attrited) / SUM(SUM(attrited)) OVER (), 1) AS share_of_attrited_pct
FROM customers
GROUP BY card_category
ORDER BY customers DESC;

-- Q9. Spend lost: attrition and lost spend by spend quintile (NTILE + CTE)
WITH q AS (
  SELECT *, NTILE(5) OVER (ORDER BY total_trans_amt) AS spend_quintile
  FROM customers
)
SELECT spend_quintile,
       MIN(total_trans_amt)                            AS min_spend,
       MAX(total_trans_amt)                            AS max_spend,
       ROUND(100.0 * AVG(attrited), 1)                 AS attrition_rate_pct,
       SUM(CASE WHEN attrited = 1 THEN total_trans_amt ELSE 0 END) AS spend_lost
FROM q
GROUP BY spend_quintile
ORDER BY spend_quintile;

-- Q10. Income category ranked by attrition rate (RANK)
SELECT income_category,
       COUNT(*)                                        AS customers,
       ROUND(100.0 * AVG(attrited), 1)                 AS attrition_rate_pct,
       RANK() OVER (ORDER BY AVG(attrited) DESC)       AS risk_rank
FROM customers
GROUP BY income_category;
