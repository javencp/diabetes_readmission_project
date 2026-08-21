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

## Project structure

```
diabetes_readmission_project/
├── data/                    # raw CSVs (UCI dataset + ID mapping lookups)
├── db/                      # SQLite database (built from the scripts below)
├── sql/
│   ├── schema.sql           # table definitions based on raw CSVs
│   ├── load_data.sql        # Cleans and loads data into tables
│   └── feature.sql          # further cleans data and joins data into clean feature table for machine learning model
└── python/                  # ML pipeline (in progress)
```
The three diagnosis fields (`diag_1`, `diag_2`, `diag_3`) were collapsed
from several hundred distinct ICD-9 codes into 9 clinically meaningful
categories. Below is the diagnosis categorizations derived from the research paper: 
## Key design decisions

- **SQLite** was chosen specifically for portability, the repo is
  cloneable and runnable without any database server setup.
- **Target** (readmitted within 30 days vs. not),
  the dataset has 3 targets: not readmitted, readmitted <30 days, and readmitted >30 days. 
  Since there is no penalty for a readmitted patient >30 days after discharge, readmission >30 days and non-readmitted patients were both aggregated to be the negative class. 
- **Columns dropped for high percentage of missing values:** `weight` (~97% missing),
  `payer_code` (~52%), and `medical_specialty` (~53%) were excluded from
  the feature table rather than imputed.
- **Diagnosis Categorization**: The three diagnosis fields (`diag_1`, `diag_2`, `diag_3`) were
  were aggregated from several hundred distinct ICD-9 codes into 9 clinically meaningful catecories. Below is the diagnosis category lookup table derived from the original paper: 

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

## How to reproduce

```bash
# From the project root, with sqlite3 on PATH:
sqlite3 db/diabetic_data.db < sql/schema.sql
sqlite3 db/diabetic_data.db < sql/load_data.sql
sqlite3 db/diabetic_data.db < sql/feature.sql
```

(On Windows PowerShell, use `Get-Content sql/schema.sql | sqlite3 db/diabetic_data.db` instead of `<` redirection.)

## Status
- SQL side of the project (cleaning and transforming raw csv into a more usable format): DONE
- Python side: In-progress
  - load, preprocess, and split data
  - train baseline (logistic regression)
  - train alternative models (XGBoost, Random Forest)
  - evaluate models
