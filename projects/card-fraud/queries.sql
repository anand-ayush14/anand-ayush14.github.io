-- Card Fraud Detection: SQL analysis (SQLite)
-- Table: transactions(txn_id, time_sec, day, hour, amount, is_fraud, V1..V28)
-- time_sec = seconds since the first transaction (2 days of data), amount in EUR

-- Q1. Portfolio overview: how big is the fraud problem?
SELECT COUNT(*)                                        AS transactions,
       SUM(is_fraud)                                   AS frauds,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3)      AS fraud_rate_pct,
       ROUND(SUM(amount), 0)                           AS total_amount_eur,
       ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount END), 0) AS fraud_amount_eur,
       ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN amount END) / SUM(amount), 3) AS fraud_amount_share_pct
FROM transactions;

-- Q2. Fraud rate by hour of day (night hours are riskier)
SELECT hour,
       COUNT(*)                                        AS transactions,
       SUM(is_fraud)                                   AS frauds,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3)      AS fraud_rate_pct
FROM transactions
GROUP BY hour
ORDER BY hour;

-- Q3. Fraud rate by transaction amount band
SELECT CASE WHEN amount = 0     THEN '1. EUR 0 (card test)'
            WHEN amount < 10    THEN '2. EUR 0-10'
            WHEN amount < 50    THEN '3. EUR 10-50'
            WHEN amount < 200   THEN '4. EUR 50-200'
            WHEN amount < 1000  THEN '5. EUR 200-1K'
            ELSE                     '6. EUR 1K+' END  AS amount_band,
       COUNT(*)                                        AS transactions,
       SUM(is_fraud)                                   AS frauds,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3)      AS fraud_rate_pct,
       ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 0) AS fraud_amount_eur
FROM transactions
GROUP BY amount_band
ORDER BY amount_band;

-- Q4. Fraud vs genuine: amount profile
SELECT CASE is_fraud WHEN 1 THEN 'Fraud' ELSE 'Genuine' END AS type,
       COUNT(*)                                        AS transactions,
       ROUND(AVG(amount), 2)                           AS avg_amount,
       ROUND(MAX(amount), 2)                           AS max_amount,
       ROUND(100.0 * AVG(CASE WHEN amount < 2 THEN 1.0 ELSE 0 END), 1) AS pct_under_2_eur
FROM transactions
GROUP BY is_fraud;

-- Q5. Night (00:00-05:59) vs day
SELECT CASE WHEN hour < 6 THEN 'Night 00-06' ELSE 'Day 06-24' END AS period,
       COUNT(*)                                        AS transactions,
       SUM(is_fraud)                                   AS frauds,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3)      AS fraud_rate_pct
FROM transactions
GROUP BY period;

-- Q6. Hourly timeline with running total and 3-hour moving average (window functions)
WITH hourly AS (
  SELECT CAST(time_sec / 3600 AS INTEGER)              AS hour_index,
         COUNT(*)                                      AS transactions,
         SUM(is_fraud)                                 AS frauds
  FROM transactions
  GROUP BY hour_index
)
SELECT hour_index,
       transactions,
       frauds,
       SUM(frauds) OVER (ORDER BY hour_index)          AS running_frauds,
       ROUND(AVG(frauds) OVER (ORDER BY hour_index
             ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS frauds_3h_avg
FROM hourly
ORDER BY hour_index;

-- Q7. Ten largest fraud transactions (RANK)
SELECT *
FROM (SELECT txn_id, day, hour, amount,
             RANK() OVER (ORDER BY amount DESC)        AS amount_rank
      FROM transactions
      WHERE is_fraud = 1)
WHERE amount_rank <= 10;

-- Q8. Fraud bursts: gap to the previous fraud (LAG). Fraudsters often hit in quick succession
WITH frauds AS (
  SELECT txn_id, time_sec,
         time_sec - LAG(time_sec) OVER (ORDER BY time_sec) AS gap_sec
  FROM transactions
  WHERE is_fraud = 1
)
SELECT CASE WHEN gap_sec IS NULL THEN '0. first fraud'
            WHEN gap_sec <= 60   THEN '1. within 1 minute'
            WHEN gap_sec <= 600  THEN '2. 1-10 minutes'
            WHEN gap_sec <= 3600 THEN '3. 10-60 minutes'
            ELSE                      '4. over 1 hour' END AS gap_to_previous_fraud,
       COUNT(*)                                        AS frauds,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS share_pct
FROM frauds
GROUP BY gap_to_previous_fraud
ORDER BY gap_to_previous_fraud;

-- Q9. Day 1 vs day 2 (is the fraud rate stable over time?)
SELECT day,
       COUNT(*)                                        AS transactions,
       SUM(is_fraud)                                   AS frauds,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3)      AS fraud_rate_pct,
       ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 0) AS fraud_amount_eur
FROM transactions
GROUP BY day
ORDER BY day;
