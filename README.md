# Diabetes 30-Day Hospital Readmission Prediction

## Background
The 30-day readmission rule refers to the Hospital Readmissions Reduction Program (HRRP). 
Essentially, hospitals are financially penalized if they have an excess of readmitted patients
within 30 days. As a result, hospitals are encouraged to improve patient care and discharge plans. 

## Objectives
- Utilize SQL to manage diabetic patient hospital records.   
- Build a Python machine learning model to accurately predict whether a diabetic patient will be readmitted within 30 days of discharge. 

**Implications**: Being able to predict which patients are most likely to be readmitted will provide useful information for hospitals in specific areas to improve upon. 

This is a skills demonstration project, not a clinical research claim.

## Dataset
[Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
from the UCI Machine Learning Repository (1999–2008) of
inpatient encounters across 130 US hospitals(~101,766 encounters).

The original research paper associated with this dataset:

> Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, et al.
> Impact of HbA1c measurement on hospital readmission rates: Analysis of
> 70,000 clinical database patient records. *BioMed Res Int.* 2014;2014:781670.

## How to Reproduce

Clone the repo. The raw CSVs used below (`data/`) are included, no separate download needed.

### 1. Build the database schema

**macOS / Linux / Git Bash:**
```bash
sqlite3 db/diabetic_data.db < sql/schema.sql
```

**Windows PowerShell:**
```powershell
Get-Content sql/schema.sql | sqlite3 db/diabetic_data.db
```

### 2. Import the raw CSVs

Open an interactive SQLite session from the project root and run:

```
sqlite3 db/diabetic_data.db
.mode csv
.import data/diabetic_data.csv raw_diabetic_data
.import --skip 1 data/discharge_disposition.csv discharge_disposition
.import --skip 1 data/admission_source.csv admission_source
.import --skip 1 data/admission_type.csv admission_type
.quit
```

### 3. Clean, load, and build features

**macOS / Linux / Git Bash:**
```bash
sqlite3 db/diabetic_data.db < sql/load_data.sql
sqlite3 db/diabetic_data.db < sql/features.sql
```

**Windows PowerShell:**
```powershell
Get-Content sql/load_data.sql | sqlite3 db/diabetic_data.db
Get-Content sql/features.sql | sqlite3 db/diabetic_data.db
```

This produces the `ml_features` table (101,766 rows) that the Python pipeline reads from.

### 4. Run the ML pipeline

To install necessary Python packages as well as internal src/ packages run the commands below: 

```bash
pip install -r requirements.txt
pip install -e .
python run_pipeline.py
```

This trains the final logistic regression and Random Forest models and saves them to `models/`.
Notebooks in `python/notebooks/` walk through the full EDA, tuning process, and model comparison.

## Results

| Model | Test ROC-AUC | Test PR-AUC | Recall @ threshold=0.45 | Precision @ threshold=0.45 |
|---|---|---|---|---|
| Logistic Regression | 0.665 | 0.227 | 0.682 | 0.163 |
| Random Forest | 0.671 | 0.238 | 0.727 | 0.161 |

Random Forest outperforms logistic regression on every metric, though the gap
is modest. Both models agree that `number_inpatient` (prior inpatient visits) is the
strongest predictor of 30-day readmission, consistent with relationship found during EDA.

**Best Performing Model**: Random Forest at threshold≈0.45, prioritizing recall
(catching ~73% of true 30-day readmissions) given the HRRP framing, at a
precision of ~0.16.

## Key Design Decisions

- **Why SQL**: much of this pipeline could be done in pandas alone. SQL was used deliberately here as part of this project's goal to demonstrate combined SQL + Python/ML skills. Also separating the data into respective tables (encounters, patients, etc.) emulates how a hospital would store their records and data i.e. as related tables, not a single flat file. Building an SQL database also produces a file that can be opened and queried by anyone with any SQL client. 
- **SQLite** was chosen specifically for portability, the repo is cloneable and runnable without any database server setup.
- **Target** (readmitted within 30 days vs. not),
  the dataset has 3 targets: not readmitted, readmitted <30 days, and readmitted >30 days. 
  Since there is no penalty for a readmitted patient >30 days after discharge, readmission >30 days and non-readmitted patients were both aggregated to be the negative class. 
- **Columns dropped for high percentage of missing values:** `weight` (~97% missing),
  `payer_code` (~52%), and `medical_specialty` (~53%) were excluded from
  the feature table rather than imputed.
- **Diagnosis Categorization**: The three diagnosis fields (`diag_1`, `diag_2`, `diag_3`) were
  aggregated from several hundred distinct ICD-9 codes into 9 clinically meaningful categories. Below is the diagnosis category lookup table derived from the original paper: 

| Category | ICD-9 code ranges |
|---|---|
| Diabetes | 250.xx |
| Circulatory | 390–459, 785 |
| Respiratory | 460–519, 786 |
| Digestive | 520–579, 787 |
| Genitourinary | 580–629, 788 |
| Injury | 800–999 |
| Musculoskeletal | 710–739 |
| Neoplasms | 140–239 |
| Other | everything else (including V/E supplemental codes) |

- **Patient-level train/test split**: to prevent leakage from patients with multiple encounters appearing in both datasets.
- **PR-AUC as primary tuning metric**: due to the class imbalance, where accuracy and ROC-AUC would risk being overly optimistic. Recall served as a secondary metric. 
- **One-hot encoding was performed on categorical data**: logistic regression requires numerical features. Kept consistent for all models so that models are directly comparable. 

## Future Work

- **XGBoost** — considered as a third model given its typically strong performance on tabular data.
- **Alternative class imbalance handling** — both models used `class_weight='balanced'`; resampling approaches (e.g., SMOTE, undersampling) are other methods of handling imbalance, and could be compared against the current weighting-based approach.