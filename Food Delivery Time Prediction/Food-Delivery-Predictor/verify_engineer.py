"""
verify_engineer.py
------------------
Verification script for DeliveryPreprocessor.
Checks that Distance_km and is_rush_hour columns exist and have no nulls.
"""
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import load_data
from src.preprocessing.engineer import DeliveryPreprocessor

SEP = "=" * 60

def main():
    print(f"\n{SEP}")
    print(" VERIFICATION: DeliveryPreprocessor")
    print(SEP)

    # 1. Load raw data
    df_raw = load_data()
    print(f"\n[1] Raw data loaded: {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")

    # 2. Run preprocessor
    preprocessor = DeliveryPreprocessor(delay_threshold=40)
    df = preprocessor.fit_transform(df_raw)
    print(f"\n[2] fit_transform complete: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"    Repr -> {preprocessor!r}")

    # 3. Check new columns exist
    new_cols = ["Distance_km", "is_rush_hour", "Delivery_Status"]
    print(f"\n[3] New columns check:")
    all_ok = True
    for col in new_cols:
        exists = col in df.columns
        nulls  = int(df[col].isna().sum()) if exists else "N/A"
        status = "OK" if exists and nulls == 0 else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"    [{status}]  '{col}' | exists={exists} | null_count={nulls}")

    # 4. Sample values
    print(f"\n[4] Sample values (first 5 rows):")
    print(df[["Time_Orderd", "Distance_km", "is_rush_hour", "Time_taken(min)", "Delivery_Status"]].head().to_string())

    # 5. Column stats
    print(f"\n[5] Distance_km stats:")
    print(f"    min={df['Distance_km'].min():.4f} km  "
          f"max={df['Distance_km'].max():.4f} km  "
          f"mean={df['Distance_km'].mean():.4f} km")

    print(f"\n[6] is_rush_hour distribution:")
    print(df["is_rush_hour"].value_counts().to_string())

    print(f"\n[7] Delivery_Status distribution (0=Fast, 1=Delayed):")
    print(df["Delivery_Status"].value_counts().to_string())

    # Final verdict
    print(f"\n{SEP}")
    if all_ok:
        print(" [PASS] All checks passed. No null values in engineered columns.")
    else:
        print(" [FAIL] One or more checks failed. See output above.")
    print(SEP)
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
