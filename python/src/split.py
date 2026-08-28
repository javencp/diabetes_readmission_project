"""
split.py

Patient-level train/test split for the ml_features dataset.

Patients may have multiple encounters (rows) in the dataset. Splitting by encounters 
will likely lead to the same patient appearing in both train and test 
presenting a potential leakage risk. 

To avoid this, we split on the list of unique patient_nbr values first, then
assign every encounter for a given patient to whichever set that patient landed in.
"""

from sklearn.model_selection import train_test_split


def patient_split(df, test_size=0.2, random_state=42):
    """
    Split a DataFrame into train/test sets using a 80/20 split
    """
    # Get the unique patient IDs
    unique_patients = df["patient_nbr"].unique()

    # Split the unique patient IDs into train and test sets
    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_size,
        random_state=random_state,
    )

    # Create train and test DataFrames by filtering based on assigned splits
    train_df = df[df["patient_nbr"].isin(train_patients)].copy()
    test_df = df[df["patient_nbr"].isin(test_patients)].copy()

    # helper functions for debugging and sanity checks
    _verify_no_patient_overlap(train_df, test_df)
    _print_split_summary(df, train_df, test_df)

    return train_df, test_df


def _verify_no_patient_overlap(train_df, test_df):
    """
    Hard-fail if any patient somehow ended up in both sets.
    """

    train_patients = set(train_df["patient_nbr"])
    test_patients = set(test_df["patient_nbr"])
    overlap = train_patients & test_patients

    assert len(overlap) == 0, (
        f"Leakage detected: {len(overlap)} patients appear in both train and test sets."
    )


def _print_split_summary(df, train_df, test_df):
    """
    Print summary statistics about the train/test split for sanity checks.
    """
    # Ensure that the train/test split is roughly 80/20 by encounters
    print(f"Total encounters: {len(df):,} | "
          f"Train: {len(train_df):,} ({len(train_df) / len(df):.1%}) | "
          f"Test: {len(test_df):,} ({len(test_df) / len(df):.1%})")

    # Check positive class imbalance in train/test sets
    if "readmitted_30" in df.columns:
        print(f"Readmit rate - Full: {df['readmitted_30'].mean():.2%} | "
              f"Train: {train_df['readmitted_30'].mean():.2%} | "
              f"Test: {test_df['readmitted_30'].mean():.2%}")

    print()


if __name__ == "__main__":
    from src.data_loader import load_features
    from src.preprocessing import clean_data

    raw = load_features()
    cleaned = clean_data(raw)
    train, test = patient_split(cleaned)
    print(train.head())