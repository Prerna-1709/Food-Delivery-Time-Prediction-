"""
data_loader.py
--------------
Utility script for the Food-Delivery-Predictor project.

Loads the raw dataset and performs a basic sanity check:
  - Prints dataset shape
  - Prints column names and data types
  - Prints the first 5 rows (head)
  - Prints full DataFrame info
"""

import os
import pandas as pd


# ── Path Configuration ────────────────────────────────────────────────────────
# Resolve the path relative to this file so the script works regardless of
# the working directory from which it is invoked.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Food_Delivery_Time_Prediction.csv")


def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV dataset from *filepath* and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at:\n  {filepath}\n"
            "Please place 'Food_Delivery_Time_Prediction.csv' inside data/raw/ and try again."
        )

    df = pd.read_csv(filepath)
    return df


def sanity_check(df: pd.DataFrame) -> None:
    """Print a basic sanity report for the given DataFrame.

    Checks performed
    ----------------
    1. Dataset shape (rows × columns)
    2. Column names and their inferred data types
    3. First 5 rows (head)
    4. Full DataFrame info (non-null counts, dtypes, memory usage)

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    """
    separator = "=" * 60

    print(separator)
    print(" FOOD DELIVERY TIME PREDICTION — DATASET SANITY CHECK")
    print(separator)

    # 1. Shape
    print(f"\n[Shape] {df.shape[0]} rows x {df.shape[1]} columns\n")

    # 2. Columns & Data Types
    print("[Columns & Data Types]")
    print("-" * 40)
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype":  df.dtypes.values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
    })
    print(dtype_df.to_string(index=False))

    # 3. Head
    print("\n[First 5 Rows (head)]")
    print("-" * 40)
    print(df.head().to_string())

    # 4. Full Info
    print("\n[DataFrame Info]")
    print("-" * 40)
    df.info()

    print(f"\n{separator}")
    print(" [OK] Sanity check complete -- no errors detected.")
    print(separator)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n[Loading] Dataset from:\n   {DATA_PATH}\n")
    df = load_data()
    sanity_check(df)
