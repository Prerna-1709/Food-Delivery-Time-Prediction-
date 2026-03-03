"""verify_processor.py — Test script for DataProcessor.

Checks:
  1. No null values remain in the encoded output.
  2. Feature count matches expectations.
  3. Distance_km and is_rush_hour columns created correctly.
  4. Delivery_Status distribution is sane.
  5. Numeric features are scaled (mean ≈ 0).
  6. Categorical features are one-hot encoded (all values 0 or 1).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from src.utils.config import RAW_CSV, NUMERIC_FEATURES, CATEGORICAL_FEATURES, BOOL_FEATURES
from src.preprocessing.processor import DataProcessor

SEP = "=" * 60

def main():
    print(f"\n{SEP}")
    print("  VERIFICATION: DataProcessor")
    print(SEP)

    # ── Load raw data ─────────────────────────────────────────────
    raw = pd.read_csv(RAW_CSV)
    print(f"\n[1] Raw data: {raw.shape[0]} rows x {raw.shape[1]} cols")

    # ── fit_transform ─────────────────────────────────────────────
    proc = DataProcessor(delay_threshold=40)
    X_train, X_test, y_train, y_test = proc.fit_transform(raw)
    print(f"\n[2] fit_transform complete: {proc!r}")
    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")

    all_ok = True
    checks = []

    # ── Check 1: No nulls in X ────────────────────────────────────
    full_X = pd.concat([X_train, X_test])
    null_count = full_X.isna().sum().sum()
    ok = null_count == 0
    checks.append(("[NULL CHECK]    No null values in encoded X",
                   ok, f"null_count={null_count}"))
    all_ok &= ok

    # ── Check 2: Feature count ────────────────────────────────────
    expected_num  = len(NUMERIC_FEATURES)
    expected_cat  = len(proc.encoder_.get_feature_names_out(CATEGORICAL_FEATURES))
    expected_bool = len(BOOL_FEATURES)
    expected_total = expected_num + expected_cat + expected_bool
    actual_total   = full_X.shape[1]
    ok = actual_total == expected_total
    checks.append(("[FEATURE COUNT] Encoded column count matches",
                   ok,
                   f"expected={expected_total} ({expected_num} num + "
                   f"{expected_cat} OHE + {expected_bool} bool), got={actual_total}"))
    all_ok &= ok

    # ── Check 3: Distance_km and is_rush_hour exist in output ─────
    for col in ["Distance_km", "is_rush_hour"]:
        ok = col in full_X.columns
        checks.append((f"[COLUMN EXISTS] '{col}' present in output", ok, ""))
        all_ok &= ok

    # ── Check 4: Delivery_Status distribution ────────────────────
    full_y = pd.concat([y_train, y_test])
    n_fast    = int((full_y == 0).sum())
    n_delayed = int((full_y == 1).sum())
    ok = n_fast > 0 and n_delayed > 0
    checks.append(("[TARGET DIST]   Both classes present",
                   ok, f"Fast={n_fast}, Delayed={n_delayed}"))
    all_ok &= ok

    # ── Check 5: Numeric features are scaled ─────────────────────
    num_means = full_X[NUMERIC_FEATURES].mean()
    ok = (num_means.abs() < 1.0).all()   # scaled mean should be near 0
    checks.append(("[SCALE CHECK]   Numeric features are StandardScaled",
                   ok, f"max |mean|={num_means.abs().max():.4f}"))
    all_ok &= ok

    # ── Check 6: OHE columns are 0/1 binary ──────────────────────
    ohe_cols = proc.encoder_.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    unique_vals = set(full_X[ohe_cols].values.ravel().tolist())
    ok = unique_vals.issubset({0.0, 1.0})
    checks.append(("[OHE CHECK]     Categorical cols are binary 0/1",
                   ok, f"unique values={sorted(unique_vals)[:6]}"))
    all_ok &= ok

    # ── Print results ─────────────────────────────────────────────
    print(f"\n[3] Verification Results:")
    print("    " + "-" * 56)
    for desc, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"    [{status}]  {desc}{suffix}")

    # ── Feature name snapshot ─────────────────────────────────────
    print(f"\n[4] Output feature names ({len(proc.feature_names_)} total):")
    for i, name in enumerate(proc.feature_names_):
        print(f"    {i+1:>3}. {name}")

    # ── Final verdict ─────────────────────────────────────────────
    print(f"\n{SEP}")
    if all_ok:
        print("  [PASS] All checks passed.")
    else:
        print("  [FAIL] One or more checks failed.")
    print(SEP + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
