-- Builds the cleaned training table with a deterministic 80/10/10 split.
-- Run: bq query --use_legacy_sql=false < sql/01_prepare_data.sql
CREATE OR REPLACE TABLE mlops.income_training AS
WITH clean AS (
  SELECT
    age,
    TRIM(workclass)      AS workclass,
    TRIM(education)      AS education,
    education_num,
    TRIM(marital_status) AS marital_status,
    TRIM(occupation)     AS occupation,
    TRIM(relationship)   AS relationship,
    TRIM(race)           AS race,
    TRIM(sex)            AS sex,
    capital_gain,
    capital_loss,
    hours_per_week,
    TRIM(native_country) AS native_country,
    IF(TRIM(income_bracket) = '>50K', 1, 0) AS high_income
  FROM `bigquery-public-data.ml_datasets.census_adult_income`
  WHERE TRIM(workclass) != '?'
    AND TRIM(occupation) != '?'
)
SELECT
  *,
  -- Hash-based split: the same row always lands in the same split.
  CASE
    WHEN MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(clean))), 100) < 80 THEN 'TRAIN'
    WHEN MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(clean))), 100) < 90 THEN 'VALIDATE'
    ELSE 'TEST'
  END AS split
FROM clean;
