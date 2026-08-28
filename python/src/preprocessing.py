"""
preprocessing.py

Cleaning and encoding logic for the ml_features table.

Split into two stages, run in this order relative to split.py:
1. clean_data() - run on the full dataset. Drops rows that 
shouldn't be in the modeling population, and fixes missing values. 
2. encode_categoricals()  - apply one-hot encoding to categorical columns 
after split, the encoder is fit on the train set and applied to both train and test sets.  
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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

# Stage 1: Clean the data before splitting into train/test sets
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

    print(f"Raw data: {start_rows:,} encounters; Cleaned data: {len(df):,} encounters "
          f"({start_rows - len(df):,} dropped total).")
    print()
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

# Stage 2: Encode categorical columns after train/test split
NON_FEATURE_COLUMNS = ["encounter_id", "patient_nbr", "readmitted_30"]


def encode_categoricals(train_df, test_df):
    """
    One-hot encode categorical columns, fit on train only, applied to both.
    """

    categorical_cols = _get_categorical_columns(train_df)
    print(f"Encoding {len(categorical_cols)} categorical columns: {categorical_cols}")
    print()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[categorical_cols])

    # Apply the fitted encoder to both train and test sets
    train_encoded = _apply_encoding(train_df, encoder, categorical_cols)
    test_encoded = _apply_encoding(test_df, encoder, categorical_cols)

    print(f"Columns before encoding: {train_df.shape[1]} | after: {train_encoded.shape[1]}")
    print()
    return train_encoded, test_encoded


def _get_categorical_columns(df):
    """
    Get a list of categorical columns/features in the DataFrame, excluding non-feature columns.
    """
    categorical = df.select_dtypes(include=["object", "string"]).columns.tolist()
    return [col for col in categorical if col not in NON_FEATURE_COLUMNS]


def _apply_encoding(df, encoder, categorical_cols):
    """
    Apply the fitted OneHotEncoder to a DataFrame and return the transformed DataFrame.
    """

    encoded_array = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols) # rename column names after encoding
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    # Drop the original categorical columns and concatenate the encoded columns
    passthrough_df = df.drop(columns=categorical_cols)
    return pd.concat([passthrough_df, encoded_df], axis=1)

if __name__ == "__main__":
    # import the necessary functions from other modules to load data and split data
    from src.data_loader import load_features
    from src.split import patient_split

    raw = load_features()
    cleaned = clean_data(raw)
    train, test = patient_split(cleaned)
    train_encoded, test_encoded = encode_categoricals(train, test)
    print(train_encoded.head())