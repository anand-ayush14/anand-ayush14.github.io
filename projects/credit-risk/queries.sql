-- =====================================================================
-- Credit Card Default Risk Analysis - SQL queries (SQLite)
-- Table: clients  (one row per credit card holder, 30,000 rows)
-- Source: UCI "Default of Credit Card Clients" (Taiwan, Apr-Sep 2005)
-- Amounts are in NT$ (New Taiwan Dollars).
-- Columns used:
--   id, limit_bal, sex, education, marriage, age,
--   pay_0 (repayment status in Sep: -2/-1/0 = paid duly, 1..8 = months late),
--   bill_amt1 (Sep bill), pay_amt1 (Sep payment), defaulted (1 = defaulted next month)
-- =====================================================================

-- Q1. Portfolio overview: customers, defaults, default rate, average limit
SELECT COUNT(*)                                   AS customers,
       SUM(defaulted)                             AS defaulters,
       ROUND(100.0 * AVG(defaulted), 2)           AS default_rate_pct,
       ROUND(AVG(limit_bal), 0)                   AS avg_credit_limit
FROM clients;

-- Q2. Default rate by most recent repayment status (strongest single signal)
SELECT CASE WHEN pay_0 <= 0 THEN 'Paid on time'
            WHEN pay_0 = 1  THEN '1 month late'
            WHEN pay_0 = 2  THEN '2 months late'
            ELSE '3+ months late' END             AS repayment_status,
       COUNT(*)                                   AS customers,
       ROUND(100.0 * AVG(defaulted), 1)           AS default_rate_pct
FROM clients
GROUP BY repayment_status
ORDER BY default_rate_pct;

-- Q3. Default rate by credit limit band
SELECT CASE WHEN limit_bal <  50000  THEN '1. < 50K'
            WHEN limit_bal < 100000  THEN '2. 50K-100K'
            WHEN limit_bal < 200000  THEN '3. 100K-200K'
            WHEN limit_bal < 400000  THEN '4. 200K-400K'
            ELSE '5. 400K+' END                   AS limit_band,
       COUNT(*)                                   AS customers,
       ROUND(100.0 * AVG(defaulted), 1)           AS default_rate_pct
FROM clients
GROUP BY limit_band
ORDER BY limit_band;

-- Q4. Credit utilisation (Sep bill / limit) vs default
SELECT CASE WHEN 1.0 * bill_amt1 / limit_bal < 0.30 THEN '1. Under 30%'
            WHEN 1.0 * bill_amt1 / limit_bal < 0.70 THEN '2. 30-70%'
            WHEN 1.0 * bill_amt1 / limit_bal < 1.00 THEN '3. 70-100%'
            ELSE '4. Over limit' END              AS utilisation_band,
       COUNT(*)                                   AS customers,
       ROUND(100.0 * AVG(defaulted), 1)           AS default_rate_pct
FROM clients
GROUP BY utilisation_band
ORDER BY utilisation_band;

-- Q5. Age band x default (window function: share of all customers)
SELECT age_band,
       customers,
       default_rate_pct,
       ROUND(100.0 * customers / SUM(customers) OVER (), 1) AS share_of_portfolio_pct
FROM (
    SELECT CASE WHEN age < 25 THEN '1. Under 25'
                WHEN age < 35 THEN '2. 25-34'
                WHEN age < 45 THEN '3. 35-44'
                WHEN age < 55 THEN '4. 45-54'
                ELSE '5. 55+' END                 AS age_band,
           COUNT(*)                               AS customers,
           ROUND(100.0 * AVG(defaulted), 1)       AS default_rate_pct
    FROM clients
    GROUP BY age_band
) t
ORDER BY age_band;

-- Q6. Education level x default (codes 4,5,6,0 grouped as "Other/unknown")
SELECT CASE education WHEN 1 THEN 'Graduate school'
                      WHEN 2 THEN 'University'
                      WHEN 3 THEN 'High school'
                      ELSE 'Other/unknown' END   AS education_level,
       COUNT(*)                                   AS customers,
       ROUND(100.0 * AVG(defaulted), 1)           AS default_rate_pct
FROM clients
GROUP BY education_level
ORDER BY default_rate_pct DESC;

-- Q7. Customers who paid NOTHING in September despite a positive bill
SELECT COUNT(*)                                   AS zero_payment_customers,
       ROUND(100.0 * AVG(defaulted), 1)           AS default_rate_pct
FROM clients
WHERE bill_amt1 > 0 AND pay_amt1 = 0;

-- Q8. Top 10 highest-exposure customers currently 2+ months late
--     (exposure = outstanding September bill)
SELECT id, limit_bal, bill_amt1 AS exposure, pay_0 AS months_late,
       RANK() OVER (ORDER BY bill_amt1 DESC) AS exposure_rank
FROM clients
WHERE pay_0 >= 2
ORDER BY exposure DESC
LIMIT 10;
