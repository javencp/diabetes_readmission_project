-- load_data.sql
-- Cleans and loads data from raw_diabetic_data (flat CSV import)
-- into the patients / encounters / diagnoses tables defined in the schema.sql file.
-- Notes on cleaning applied here:
--   - '?' is this dataset's missing-value marker -> converted to NULL
--   - All raw_diabetic_data columns are TEXT (CSV import), so
--     numeric fields are explicitly CAST to INTEGER
-- Run order matters: patients -> encounters -> diagnoses,
-- since encounters references patients, and diagnoses
-- references encounters (foreign keys).

PRAGMA foreign_keys = ON;

BEGIN TRANSACTION; -- ensures that all inserts succeed or fail together

-- patients table
-- One row per unique patient_nbr. Some patients have multiple
-- encounters with identical race/gender, so we just take the
-- first occurrence (by lowest encounter_id) per patient.

INSERT INTO patients (patient_nbr, race, gender)
SELECT patient_nbr, race, gender
FROM (
    SELECT
        CAST(patient_nbr AS INTEGER) AS patient_nbr,
        CASE 
            WHEN race = '?' THEN NULL 
            ELSE race 
        END AS race,
        CASE 
            WHEN gender = '?' OR gender = 'Unknown/Invalid' THEN NULL 
            ELSE gender 
        END AS gender,
        ROW_NUMBER() OVER (
            PARTITION BY patient_nbr
            ORDER BY CAST(encounter_id AS INTEGER)
        ) AS rn
    FROM raw_diabetic_data
)
WHERE rn = 1;

-- encounters
-- One row per encounter_id. Numeric fields cast from TEXT.
-- '?' cleaned to NULL for weight, payer_code, medical_specialty.
-- hyphenated raw column names (e.g. glyburide-metformin) are
-- quoted here and mapped to their underscore equivalents.

INSERT INTO encounters (
    encounter_id, patient_nbr, age, weight,
    admission_type_id, discharge_disposition_id, admission_source_id,
    time_in_hospital, payer_code, medical_specialty,
    num_lab_procedures, num_procedures, num_medications,
    number_outpatient, number_emergency, number_inpatient, number_diagnoses,
    max_glu_serum, A1Cresult,
    metformin, repaglinide, nateglinide, chlorpropamide, glimepiride,
    acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone,
    rosiglitazone, acarbose, miglitol, troglitazone, tolazamide,
    examide, citoglipton, insulin,
    glyburide_metformin, glipizide_metformin, glimepiride_pioglitazone,
    metformin_rosiglitazone, metformin_pioglitazone,
    change, diabetesMed, readmitted
)
SELECT
    CAST(encounter_id AS INTEGER),
    CAST(patient_nbr AS INTEGER),
    age,
    CASE 
        WHEN weight = '?' THEN NULL 
        ELSE weight 
    END,
    CAST(admission_type_id AS INTEGER),
    CAST(discharge_disposition_id AS INTEGER),
    CAST(admission_source_id AS INTEGER),
    CAST(time_in_hospital AS INTEGER),
    CASE 
        WHEN payer_code = '?' THEN NULL 
        ELSE payer_code 
    END,
    CASE 
        WHEN medical_specialty = '?' THEN NULL 
        ELSE medical_specialty 
    END,
    CAST(num_lab_procedures AS INTEGER),
    CAST(num_procedures AS INTEGER),
    CAST(num_medications AS INTEGER),
    CAST(number_outpatient AS INTEGER),
    CAST(number_emergency AS INTEGER),
    CAST(number_inpatient AS INTEGER),
    CAST(number_diagnoses AS INTEGER),
    max_glu_serum,
    A1Cresult,
    metformin, repaglinide, nateglinide, chlorpropamide, glimepiride,
    acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone,
    rosiglitazone, acarbose, miglitol, troglitazone, tolazamide,
    examide, citoglipton, insulin,
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
    change, diabetesMed, readmitted
FROM raw_diabetic_data;

-- diagnoses
-- Unpivoted from diag_1 / diag_2 / diag_3. Rows with '?' or
-- blank codes are skipped (no diagnosis recorded in that slot).
-- diagnosis_rank preserves whether it was primary (1),
-- secondary (2), or additional (3).

INSERT INTO diagnoses (encounter_id, diagnosis_code, diagnosis_rank)
SELECT CAST(encounter_id AS INTEGER), diag_1, 1
FROM raw_diabetic_data
WHERE diag_1 IS NOT NULL AND diag_1 != '?'

UNION ALL

SELECT CAST(encounter_id AS INTEGER), diag_2, 2
FROM raw_diabetic_data
WHERE diag_2 IS NOT NULL AND diag_2 != '?'

UNION ALL

SELECT CAST(encounter_id AS INTEGER), diag_3, 3
FROM raw_diabetic_data
WHERE diag_3 IS NOT NULL AND diag_3 != '?';

COMMIT;