-- data_analysis.sql
-- Initial exploratory data analysis (EDA) performed on the data 
-- that is naturally more suited for SQL than Python. 
-- Here are some of the most interesting query results 
-- analyzing the ml_features table (built by features.sql script).
--
-- Each query below is executed and results can be visualized in the 
-- python/notebooks/01_eda.ipynb script, but are included here as the standalone SQL reference.


-- Rate of readmission within 30 days, grouped by number of prior inpatient visits
SELECT
    CASE
        WHEN number_inpatient = 0 THEN '0'
        WHEN number_inpatient = 1 THEN '1'
        WHEN number_inpatient = 2 THEN '2'
        ELSE '3+'
    END AS prior_inpatient_visits,
    COUNT(*) AS encounters,
    ROUND(AVG(readmitted_30) * 100, 2) AS readmit_rate_pct
FROM ml_features
-- WHERE clause emulates preprocessing performed in python/src/preprocessing.py to 
-- filter out encounters that are not relevant for readmission analysis
WHERE discharge_disposition NOT IN (
    'Expired', 
    'Hospice / home', 
    'Hospice / medical facility',
    'Expired at home. Medicaid only, hospice.',
    'Expired in a medical facility. Medicaid only, hospice.', 
    'Expired, place unknown. Medicaid only, hospice.' 
) AND gender IS NOT NULL AND gender != 'Unknown/Invalid'
GROUP BY prior_inpatient_visits
ORDER BY prior_inpatient_visits;

-- Rate of readmission within 30 days, grouped by primary diagnosis category
-- Essentially the same as the previous query, but grouped by a different feature
SELECT
    primary_diagnosis_category,
    COUNT(*) AS encounters,
    ROUND(AVG(readmitted_30) * 100, 2) AS readmit_rate_pct
FROM ml_features
WHERE discharge_disposition NOT IN (
    'Expired', 
    'Hospice / home', 
    'Hospice / medical facility',
    'Expired at home. Medicaid only, hospice.',
    'Expired in a medical facility. Medicaid only, hospice.',
    'Expired, place unknown. Medicaid only, hospice.' 
) AND gender IS NOT NULL AND gender != 'Unknown/Invalid'
GROUP BY primary_diagnosis_category
ORDER BY readmit_rate_pct DESC;


-- Rate of readmission within 30 days, grouped by discharge disposition
SELECT
    discharge_disposition,
    COUNT(*) AS encounters,
    ROUND(AVG(readmitted_30) * 100, 2) AS readmit_rate_pct
FROM ml_features
WHERE discharge_disposition NOT IN (
    'Expired', 
    'Hospice / home', 
    'Hospice / medical facility',
    'Expired at home. Medicaid only, hospice.',
    'Expired in a medical facility. Medicaid only, hospice.',
    'Expired, place unknown. Medicaid only, hospice.' 
) AND gender IS NOT NULL AND gender != 'Unknown/Invalid'
GROUP BY discharge_disposition
-- filter out small sample sizes to avoid misleading results
HAVING COUNT(*) > 100
ORDER BY readmit_rate_pct DESC;


-- Rate of readmission within 30 days, grouped by HbA1c test result
-- (Strack et al. 2014 found A1C testing associated with lower readmission -
-- testing whether that pattern holds in this cleaned population)
SELECT
    A1Cresult,
    COUNT(*) AS encounters,
    ROUND(AVG(readmitted_30) * 100, 2) AS readmit_rate_pct
FROM ml_features
WHERE discharge_disposition NOT IN (
    'Expired', 
    'Hospice / home', 
    'Hospice / medical facility',
    'Expired at home. Medicaid only, hospice.',
    'Expired in a medical facility. Medicaid only, hospice.',
    'Expired, place unknown. Medicaid only, hospice.' 
) AND gender IS NOT NULL AND gender != 'Unknown/Invalid'
GROUP BY A1Cresult
ORDER BY readmit_rate_pct DESC;