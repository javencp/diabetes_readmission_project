"""
data_loader.py

Connects to the project's SQLite database and loads the final feature table produced by features.sql file
"""

import sqlite3
from pathlib import Path
import pandas as pd


default_db_path = Path("db/diabetic_data.db")
feature_table = "ml_features"


def load_features(db_path=default_db_path, table_name=feature_table):
    """
    Inputs = Path, str
    Outputs = pd.DataFrame
    Load the final feature table from the SQLite database into a dataframe.
    """

    # Check that the database file exists
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path.resolve()}. "
            "Check that you're running from the project root."
        )

    # Connect to the SQLite database and load the feature table into a pandas DataFrame
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()

    _run_sanity_checks(df)
    return df


def _run_sanity_checks(df):
    """
    Print basic diagnostics right after load to catch SQL/Python handoff issues early
    """

    # Print number of rows and columns
    # Should be 101,766 encounters/rows expected from the SQL query
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")

    n_patients = df["patient_nbr"].nunique()
    print(f"Unique patients: {n_patients:,}")

    # detect missing values in the dataframe and print columns with missing values
    null_counts = df.isnull().sum()
    nonzero_nulls = null_counts[null_counts > 0]
    if len(nonzero_nulls) > 0:
        print("Columns with missing values:")
        print(nonzero_nulls.to_string())
    else:
        print("No missing values detected.")

    # calculates and prints the percentage of the positive class
    # i.e. the readmission rate within 30 days
    if "readmitted_30" in df.columns:
        rate = df["readmitted_30"].mean()
        print(f"Readmitted-within-30-days rate: {rate:.2%}")


if __name__ == "__main__":
    data = load_features()
    print(data.head())