"""
preprocessing.py

Cleaning and encoding logic for the ml_features table.

Data workflow: 
1. clean_data() - (in preprocessing.py) run on the full dataset. Drops rows that 
    shouldn't be in the modeling population, and fixes missing values. 
2. patient_split() - (in split.py) run on the cleaned DataFrame, 
    returning train and test DataFrames. 
3. get_features_and_target() - (in split.py) is run on the train and test DataFrames to 
    produce feature matrices and target vectors.
4. encode_categoricals() - (in preprocessing.py) performs one-hot encoding on the categorical features
    in the feature matrices
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

    print(f"\nRaw data: {start_rows:,} encounters; Cleaned data: {len(df):,} encounters "
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

# Stage 2: Encode categorical columns after train/test split
def encode_categoricals(X_train, X_test):
    """
    One-hot encode categorical columns, fit on X_train only, applied to both.
    This keeps train/test columns guaranteed identical.
 
    Input: X_train, X_test (both pd.DataFrame)
        Expects the feature matrices which should be 
        the output of split.get_features_and_target().
 
    Output: X_train_encoded, X_test_encoded (both pd.DataFrame)
        Same rows as input, but with categorical columns one-hot encoded 
        and original categorical columns dropped.
    """

    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"Encoding {len(categorical_cols)} categorical columns: {categorical_cols}")
 
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[categorical_cols])

     # Apply the fitted encoder to both train and test feature matrices
    X_train_encoded = _apply_encoding(X_train, encoder, categorical_cols)
    X_test_encoded = _apply_encoding(X_test, encoder, categorical_cols)
 
    print(f"\nColumns before encoding: {X_train.shape[1]} | after: {X_train_encoded.shape[1]}")
    return X_train_encoded, X_test_encoded


# Helper function to apply the fitted encoder to a DataFrame
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
    from src.data_loader import load_features
    from src.split import patient_split, get_features_and_target
 
    raw = load_features()
    cleaned = clean_data(raw)
    train, test = patient_split(cleaned)
    X_train, y_train, ids_train = get_features_and_target(train)
    X_test, y_test, ids_test = get_features_and_target(test)
    X_train_encoded, X_test_encoded = encode_categoricals(X_train, X_test)
    print(X_train_encoded.head())