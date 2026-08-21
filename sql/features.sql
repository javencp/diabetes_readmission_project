-- feature.sql
-- Builds the final ML-ready table (ml_features) by joining
-- encounters with patients and lookup tables, converting age
-- to numeric, binarizing the readmitted label, and aggregating
-- diagnosis codes into corresponding categories

-- Columns dropped entirely due to high percentage of missing vales: 
--   weight: (~97% missing)
--   payer_code: (~52% missing)
--   medical_specialty: (~53% missing)

DROP TABLE IF EXISTS ml_features; -- here for convencience, in case of troubleshooting

CREATE TABLE ml_features AS

WITH diagnosis_numeric AS (
    -- Convert diagnosis_code to REAL, except for V/E codes which are
    -- are instead mapped to NULL.
    SELECT
        encounter_id,
        diagnosis_rank,
        CASE WHEN SUBSTR(diagnosis_code, 1, 1) IN ('V', 'E')
             THEN NULL
             ELSE CAST(diagnosis_code AS REAL)
        END AS code_numeric
    FROM diagnoses
),

diagnosis_categorized AS (
    SELECT
        encounter_id,
        diagnosis_rank,
        CASE
            WHEN code_numeric IS NULL THEN 'other'
            WHEN code_numeric >= 250 AND code_numeric < 251 THEN 'diabetes'
            WHEN code_numeric >= 390 AND code_numeric < 460 THEN 'circulatory'
            WHEN code_numeric = 785 THEN 'circulatory'
            WHEN code_numeric >= 460 AND code_numeric < 520 THEN 'respiratory'
            WHEN code_numeric = 786 THEN 'respiratory'
            WHEN code_numeric >= 520 AND code_numeric < 580 THEN 'digestive'
            WHEN code_numeric = 787 THEN 'digestive'
            WHEN code_numeric >= 580 AND code_numeric < 630 THEN 'genitourinary'
            WHEN code_numeric = 788 THEN 'genitourinary'
            WHEN code_numeric >= 800 AND code_numeric < 1000 THEN 'injury'
            WHEN code_numeric >= 710 AND code_numeric < 740 THEN 'musculoskeletal'
            WHEN code_numeric >= 140 AND code_numeric < 240 THEN 'neoplasms'
            ELSE 'other'
        END AS diagnosis_category
    FROM diagnosis_numeric
),

diagnosis_features AS (
    -- determines if this category was present in any of the
    -- encounter's primary or secondary diagnoses (diag_1, diag_2, diag_3)
    SELECT
        encounter_id,
        MAX(CASE WHEN diagnosis_category = 'diabetes'    THEN 1 ELSE 0 END) AS has_diabetes_diagnosis,
        MAX(CASE WHEN diagnosis_category = 'circulatory' THEN 1 ELSE 0 END) AS has_circulatory_diagnosis,
        MAX(CASE WHEN diagnosis_category = 'respiratory' THEN 1 ELSE 0 END) AS has_respiratory_diagnosis,
        MAX(CASE WHEN diagnosis_category = 'digestive'   THEN 1 ELSE 0 END) AS has_digestive_diagnosis,
        MAX(CASE WHEN diagnosis_category = 'genitourinary' THEN 1 ELSE 0 END) AS has_genitourinary_diagnosis,
        MAX(CASE WHEN diagnosis_category = 'injury'  THEN 1 ELSE 0 END) AS has_injury_diagnosis,
        MAX(CASE WHEN diagnosis_category = 'musculoskeletal' THEN 1 ELSE 0 END) AS has_musculoskeletal_diagnosis,
        MAX(CASE WHEN diagnosis_category = 'neoplasms'  THEN 1 ELSE 0 END) AS has_neoplasms_diagnosis,
        MAX(CASE WHEN diagnosis_category = 'other' THEN 1 ELSE 0 END) AS has_other_diagnosis
    FROM diagnosis_categorized
    GROUP BY encounter_id
),

primary_diagnosis AS (
    -- finds the primary diagnosis category for each encounter
    SELECT
        encounter_id,
        diagnosis_category AS primary_diagnosis_category
    FROM diagnosis_categorized
    WHERE diagnosis_rank = 1
)

SELECT
    e.encounter_id,
    e.patient_nbr,

    -- Patient-level attributes
    p.race,
    p.gender,

    -- Age ranges converted to their numeric midpoint
    CASE e.age
        WHEN '[0-10)'   THEN 5
        WHEN '[10-20)'  THEN 15
        WHEN '[20-30)'  THEN 25
        WHEN '[30-40)'  THEN 35
        WHEN '[40-50)'  THEN 45
        WHEN '[50-60)'  THEN 55
        WHEN '[60-70)'  THEN 65
        WHEN '[70-80)'  THEN 75
        WHEN '[80-90)'  THEN 85
        WHEN '[90-100)' THEN 95
    END AS age_numeric,

    -- Lookup-decoded descriptions instead of raw numeric codes
    at.description AS admission_type,
    dd.description AS discharge_disposition,
    src.description AS admission_source,

    -- Encounter-level counts and utilization history
    e.time_in_hospital,
    e.num_lab_procedures,
    e.num_procedures,
    e.num_medications,
    e.number_outpatient,
    e.number_emergency,
    e.number_inpatient,
    e.number_diagnoses,

    -- Lab results
    e.max_glu_serum,
    e.A1Cresult,

    -- Medications
    e.metformin, e.repaglinide, e.nateglinide, e.chlorpropamide, e.glimepiride,
    e.acetohexamide, e.glipizide, e.glyburide, e.tolbutamide, e.pioglitazone,
    e.rosiglitazone, e.acarbose, e.miglitol, e.troglitazone, e.tolazamide,
    e.examide, e.citoglipton, e.insulin,
    e.glyburide_metformin, e.glipizide_metformin, e.glimepiride_pioglitazone,
    e.metformin_rosiglitazone, e.metformin_pioglitazone,
    e.change, e.diabetesMed,

    -- Diagnosis-derived features
    COALESCE(pd.primary_diagnosis_category, 'none') AS primary_diagnosis_category,
    COALESCE(df.has_diabetes_diagnosis, 0) AS has_diabetes_diagnosis,
    COALESCE(df.has_circulatory_diagnosis, 0) AS has_circulatory_diagnosis,
    COALESCE(df.has_respiratory_diagnosis, 0) AS has_respiratory_diagnosis,
    COALESCE(df.has_digestive_diagnosis, 0) AS has_digestive_diagnosis,
    COALESCE(df.has_genitourinary_diagnosis, 0) AS has_genitourinary_diagnosis,
    COALESCE(df.has_injury_diagnosis, 0) AS has_injury_diagnosis,
    COALESCE(df.has_musculoskeletal_diagnosis, 0) AS has_musculoskeletal_diagnosis,
    COALESCE(df.has_neoplasms_diagnosis, 0) AS has_neoplasms_diagnosis,
    COALESCE(df.has_other_diagnosis, 0) AS has_other_diagnosis,

    -- Target label: 1 if readmitted within 30 days, else 0
    CASE WHEN e.readmitted = '<30' THEN 1 ELSE 0 END AS readmitted_30

FROM encounters e
LEFT JOIN patients p               ON e.patient_nbr = p.patient_nbr
LEFT JOIN admission_type at        ON e.admission_type_id = at.admission_type_id
LEFT JOIN discharge_disposition dd ON e.discharge_disposition_id = dd.discharge_disposition_id
LEFT JOIN admission_source src     ON e.admission_source_id = src.admission_source_id
LEFT JOIN diagnosis_features df    ON e.encounter_id = df.encounter_id
LEFT JOIN primary_diagnosis pd     ON e.encounter_id = pd.encounter_id;