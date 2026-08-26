"""
preprocessing.py

Cleaning and encoding logic for the ml_features table.

Split into two stages, run in this order relative to split.py:
1. clean_data() - run on the full dataset. Drops rows that 
shouldn't be in the modeling population, and fixes missing values. 
2. encode_categoricals()  - apply one-hot encoding to categorical columns --> Not yet implemented. 
"""

import pandas as pd

# Discharge dispositions where the patient died are excluded from the modeling
# population entirely rather than treated as valid negative examples.
# Patients discharged to hospice are also excluded, since they are not expected to be readmitted.
# The readmission outcome is not meaningful for these patients, and including them would bias the model.
EXCLUDE_DISPOSITIONS = [
    "Expired",
    "Hospice / home",
    "Hospice / medical facility",
    "Expired at home. Medicaid only, hospice.",
    "Expired in a medical facility. Medicaid only, hospice.",
    "Expired, place unknown. Medicaid only, hospice.",
]


def clean_data(df):
    """
    Clean the raw ml_features DataFrame
    """
    df = df.copy() 
    start_rows = len(df)

    # 3 helper functions to clean the data
    df = _drop_death_hospice(df)
    df = _drop_invalid_gender(df)
    df = _fill_missing_race(df)

    print(f"clean_data: {start_rows:,} rows in, {len(df):,} rows out "
          f"({start_rows - len(df):,} dropped total).")
    return df

def _drop_death_hospice(df):
    """
    helper function to remove rows based on exclusion list defined above
    """
    before = len(df)
    df = df[~df["discharge_disposition"].isin(EXCLUDE_DISPOSITIONS)].copy()
    print(f"  Dropped {before - len(df):,} rows with death/hospice discharge disposition.")
    return df


def _drop_invalid_gender(df):
    """
    helper function to remove rows with missing or invalid gender values
    """
    before = len(df)
    df = df[df["gender"].notna()].copy() #scraps non-null vales
    df = df[df["gender"] != "Unknown/Invalid"].copy() #drops invalid gender values
    print(f"  Dropped {before - len(df):,} rows with missing/invalid gender.")
    return df


def _fill_missing_race(df):
    """
    helper function to fill missing race values with 'Unknown' category.
    """
    before_missing = df["race"].isna().sum() #counts missing values
    df = df.copy()
    df["race"] = df["race"].fillna("Unknown")
    print(f"  Filled {before_missing:,} missing race values with 'Unknown'.")
    return df


if __name__ == "__main__":
    # load the raw features from the database 
    from src.data_loader import load_features
 
    raw = load_features()
    cleaned = clean_data(raw)
    print(cleaned.head())