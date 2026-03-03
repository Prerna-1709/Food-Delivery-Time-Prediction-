"""Regression pipeline: predict exact food delivery time (in minutes)."""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Path setup – works regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.engineer import (  # noqa: E402
    COL_TIME_TAKEN,
    DeliveryPreprocessor,
)
from src.utils.data_loader import load_data  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAVE_DIR    = os.path.join(PROJECT_ROOT, "models", "saved")
MODEL_PATH  = os.path.join(SAVE_DIR, "linear_regression.joblib")
TARGET_COL  = COL_TIME_TAKEN          # "Time_taken(min)"
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# Features fed to the model after engineering
NUMERIC_FEATURES      = ["Distance_km", "Delivery_person_Age", "Delivery_person_Ratings",
                          "Vehicle_condition", "multiple_deliveries"]
CATEGORICAL_FEATURES  = ["Weather_conditions", "Road_traffic_density",
                          "Type_of_order", "Type_of_vehicle", "City", "Festival"]
BOOL_FEATURES         = ["is_rush_hour"]


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """Return a scikit-learn Pipeline with preprocessing + LinearRegression."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  StandardScaler(),
             NUMERIC_FEATURES),
            ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            # Boolean column passes through as 0/1 int – no scaling needed
            ("bool", "passthrough",
             BOOL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    LinearRegression()),
    ])


# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------
def run(df_clean: pd.DataFrame) -> dict:
    """Train a Linear Regression model and return evaluation metrics.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Feature-engineered dataframe (output of DeliveryPreprocessor).

    Returns
    -------
    dict with keys: mse, rmse, mae, r2
    """
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOL_FEATURES
    X = df_clean[all_features]
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    print(f"  Train size : {len(X_train)} rows")
    print(f"  Test size  : {len(X_test)} rows")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    return dict(pipeline=pipeline, mse=mse, rmse=rmse, mae=mae, r2=r2,
                y_test=y_test, y_pred=y_pred)


def save_model(pipeline: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Model saved  : {path}  ({size_kb:.1f} KB)")


def print_metrics(metrics: dict) -> None:
    sep = "=" * 55
    print(f"\n{sep}")
    print("  LINEAR REGRESSION — EVALUATION METRICS")
    print(sep)
    print(f"  MSE  (Mean Squared Error)  : {metrics['mse']:>10.4f}")
    print(f"  RMSE (Root MSE)            : {metrics['rmse']:>10.4f} min")
    print(f"  MAE  (Mean Absolute Error) : {metrics['mae']:>10.4f} min")
    print(f"  R2   (Coefficient of Det.) : {metrics['r2']:>10.4f}")
    print(sep)

    # Actual vs Predicted table
    print("\n  Actual vs Predicted (test set):")
    print(f"  {'Actual':>8}  {'Predicted':>10}  {'Error':>8}")
    print("  " + "-" * 32)
    for actual, pred in zip(metrics["y_test"], metrics["y_pred"]):
        print(f"  {actual:>8.1f}  {pred:>10.2f}  {actual - pred:>+8.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sep = "=" * 55

    print(f"\n{sep}")
    print("  STEP 3 — REGRESSION PIPELINE")
    print(sep)

    # 1. Load & preprocess
    print("\n[1] Loading and engineering features ...")
    raw_df = load_data()
    preprocessor = DeliveryPreprocessor()
    df = preprocessor.fit_transform(raw_df)
    print(f"    Shape after engineering: {df.shape}")

    # 2. Train
    print("\n[2] Training Linear Regression (80/20 split) ...")
    results = run(df)

    # 3. Metrics
    print_metrics(results)

    # 4. Save model
    print(f"\n[3] Saving model ...")
    save_model(results["pipeline"], MODEL_PATH)

    print(f"\n[DONE] Regression pipeline complete.\n")
