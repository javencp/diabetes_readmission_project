-- schema.sql
-- normalized schema for the Diabetes 130-US Hospitals dataset
-- source: raw_diabetic_data (flat staging table imported from CSV)

PRAGMA foreign_keys = ON;

-- Lookup tables 
-- decode the numeric ID columns from the dataset's IDS_mapping
-- reference file into human-readable descriptions
-- created for improved readability and to support joins

CREATE TABLE admission_type (
    admission_type_id   INTEGER PRIMARY KEY,
    description          TEXT
);

CREATE TABLE discharge_disposition (
    discharge_disposition_id   INTEGER PRIMARY KEY,
    description                  TEXT
);

CREATE TABLE admission_source (
    admission_source_id   INTEGER PRIMARY KEY,
    description             TEXT
);

-- patients table
-- one row per unique patient_nbr will hold only attributes that
-- are unique to the patient and unchanging across encounters

CREATE TABLE patients (
    patient_nbr   INTEGER PRIMARY KEY,
    race          TEXT,
    gender        TEXT
);

-- encounters table (which will contain the bulk of the dataset)
-- one row per encounter_id (hospital visit). Everything that
-- can vary visit-to-visit lives here, including age (treated
-- as a snapshot at time of encounter, not a patient constant)

CREATE TABLE encounters (
    encounter_id                    INTEGER PRIMARY KEY,
    patient_nbr                      INTEGER NOT NULL,
    age                               TEXT,
    weight                            TEXT,
    admission_type_id                 INTEGER,
    discharge_disposition_id          INTEGER,
    admission_source_id               INTEGER,
    time_in_hospital                  INTEGER,
    payer_code                        TEXT,
    medical_specialty                 TEXT,
    num_lab_procedures                INTEGER,
    num_procedures                    INTEGER,
    num_medications                   INTEGER,
    number_outpatient                 INTEGER,
    number_emergency                  INTEGER,
    number_inpatient                  INTEGER,
    number_diagnoses                  INTEGER,
    max_glu_serum                     TEXT,
    A1Cresult                         TEXT,

    -- Medications (each: 'No', 'Steady', 'Up', 'Down')
    metformin                         TEXT,
    repaglinide                       TEXT,
    nateglinide                       TEXT,
    chlorpropamide                    TEXT,
    glimepiride                       TEXT,
    acetohexamide                     TEXT,
    glipizide                         TEXT,
    glyburide                         TEXT,
    tolbutamide                       TEXT,
    pioglitazone                      TEXT,
    rosiglitazone                     TEXT,
    acarbose                          TEXT,
    miglitol                          TEXT,
    troglitazone                      TEXT,
    tolazamide                        TEXT,
    examide                           TEXT,
    citoglipton                       TEXT,
    insulin                           TEXT,
    glyburide_metformin               TEXT,
    glipizide_metformin               TEXT,
    glimepiride_pioglitazone          TEXT,
    metformin_rosiglitazone           TEXT,
    metformin_pioglitazone            TEXT,

    -- Medication-related summary flags
    change                             TEXT,   -- 'No' or 'Ch' (change in diabetic meds)
    diabetesMed                        TEXT,   -- 'No' or 'Yes'

    -- Target label
    readmitted                         TEXT,

    FOREIGN KEY (patient_nbr) REFERENCES patients(patient_nbr),
    FOREIGN KEY (admission_type_id) REFERENCES admission_type(admission_type_id),
    FOREIGN KEY (discharge_disposition_id) REFERENCES discharge_disposition(discharge_disposition_id),
    FOREIGN KEY (admission_source_id) REFERENCES admission_source(admission_source_id)
);

-- diagnoses table
-- Unpivoted from the diagnosis columns: diag_1/diag_2/diag_3 
-- one row per diagnosis code per encounter
-- diagnosis_rank is an added column preserving whether
-- it was the primary (1), secondary (2), or additional (3) diagnosis
-- ICD-9 codes have hundreds of distinct values, so this table may or may not 
-- be aggregated down into a handful of engineered
-- columns during feature engineering 

CREATE TABLE diagnoses (
    diagnosis_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    encounter_id      INTEGER NOT NULL,
    diagnosis_code    TEXT,
    diagnosis_rank    INTEGER,   -- 1 = diag_1, 2 = diag_2, 3 = diag_3

    FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id)
);