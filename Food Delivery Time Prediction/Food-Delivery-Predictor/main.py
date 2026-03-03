"""main.py — Unified Prediction CLI for Food-Delivery-Predictor.

Usage
-----
    python main.py --distance 5.2 --weather Sunny --traffic Low --time 18:30:00

    # or use short flags
    python main.py -d 5.2 -w Stormy -t Jam -T 08:15:00

Output
------
    Predicted Time : 34.7 mins
    Status         : Fast  (Delivery_Status = 0)

Notes
-----
    The saved pipelines (Scaler + OHE + Model) are loaded from models/saved/.
    The four inputs map directly to the most predictive features; other
    columns are filled with sensible median / mode defaults so the
    ColumnTransformer receives a complete, correctly shaped DataFrame.
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.engineer import (   # noqa: E402
    MORNING_RUSH, EVENING_RUSH,
    DeliveryPreprocessor,
)
from src.models.regression_pipeline import (   # noqa: E402
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BOOL_FEATURES,
)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
SAVE_DIR         = os.path.join(PROJECT_ROOT, "models", "saved")
REGRESSION_MODEL = os.path.join(SAVE_DIR, "linear_regression.joblib")
CLASSIFIER_MODEL = os.path.join(SAVE_DIR, "naive_bayes.joblib")   # best F1 on test set

# ---------------------------------------------------------------------------
# Default values for features not supplied by the user
# (representative medians / modes from the training data)
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "Delivery_person_Age":     30,
    "Delivery_person_Ratings": 4.6,
    "Vehicle_condition":       1,
    "multiple_deliveries":     0.0,
    "Type_of_order":           "Meal",
    "Type_of_vehicle":         "motorcycle",
    "City":                    "Urban",
    "Festival":                "No",
}

# Validation options (drawn from the CSV schema)
VALID_WEATHER = {"Sunny", "Cloudy", "Stormy", "Fog", "Windy", "Sandstorm"}
VALID_TRAFFIC = {"Low", "Medium", "High", "Jam"}

# Status labels
STATUS_LABEL = {0: "Fast", 1: "Delayed"}


# ---------------------------------------------------------------------------
# Input building
# ---------------------------------------------------------------------------
def _is_rush_hour(time_str: str) -> bool:
    """Return True if *time_str* (HH:MM:SS) falls within a rush-hour window."""
    try:
        hour = pd.to_datetime(time_str, format="%H:%M:%S").hour
        return (
            MORNING_RUSH[0] <= hour <= MORNING_RUSH[1]
            or EVENING_RUSH[0] <= hour <= EVENING_RUSH[1]
        )
    except Exception:
        return False


def build_input_row(
    distance_km: float,
    weather: str,
    traffic: str,
    time_str: str,
) -> pd.DataFrame:
    """Construct a one-row DataFrame matching the feature schema expected by the pipelines.

    Parameters
    ----------
    distance_km : float   Haversine distance (km) between restaurant and delivery.
    weather     : str     Weather condition (e.g. 'Sunny', 'Stormy').
    traffic     : str     Road traffic density (e.g. 'Low', 'High', 'Jam').
    time_str    : str     Order time in HH:MM:SS format (24-hour clock).

    Returns
    -------
    pd.DataFrame with exactly one row and all model-expected columns.
    """
    row = {
        # User-supplied
        "Distance_km":           distance_km,
        "Weather_conditions":    weather,
        "Road_traffic_density":  traffic,
        "is_rush_hour":          _is_rush_hour(time_str),
        # Defaults
        **DEFAULTS,
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} model not found at:\n  {path}\n"
            "Run the training scripts first:\n"
            "  python src/models/regression_pipeline.py\n"
            "  python src/models/classifier_suite.py"
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Food Delivery Predictor — predict delivery time and status.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --distance 5.2 --weather Sunny --traffic Low --time 18:30:00\n"
            "  python main.py -d 12.0 -w Stormy -t Jam -T 08:00:00\n"
        ),
    )
    parser.add_argument("-d", "--distance", type=float, required=True,
                        help="Distance in km between restaurant and delivery location.")
    parser.add_argument("-w", "--weather",  type=str,   required=True,
                        choices=sorted(VALID_WEATHER),
                        help="Current weather condition.")
    parser.add_argument("-t", "--traffic",  type=str,   required=True,
                        choices=sorted(VALID_TRAFFIC),
                        help="Current road traffic density.")
    parser.add_argument("-T", "--time",     type=str,   required=True,
                        metavar="HH:MM:SS",
                        help="Order placement time in 24-hour HH:MM:SS format.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(distance_km: float, weather: str, traffic: str, time_str: str) -> dict:
    """Run both models on the supplied inputs and return a result dict.

    Returns
    -------
    dict with keys: predicted_time (float), status_code (int), status_label (str),
                    is_rush_hour (bool), input_summary (dict)
    """
    # Build feature row
    X = build_input_row(distance_km, weather, traffic, time_str)
    rush = bool(X["is_rush_hour"].iloc[0])

    # Ensure columns are in the exact order the pipelines expect
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOL_FEATURES
    X_ordered = X[feature_cols]

    # Load models
    reg_model = _load(REGRESSION_MODEL, "Regression")
    clf_model = _load(CLASSIFIER_MODEL, "Classifier")

    # Predict
    predicted_time  = float(reg_model.predict(X_ordered)[0])
    predicted_class = int(clf_model.predict(X_ordered)[0])

    return {
        "predicted_time": round(predicted_time, 1),
        "status_code":    predicted_class,
        "status_label":   STATUS_LABEL[predicted_class],
        "is_rush_hour":   rush,
        "input_summary":  {
            "Distance km":   distance_km,
            "Weather":       weather,
            "Traffic":       traffic,
            "Order time":    time_str,
            "Rush hour":     rush,
        },
    }


def print_result(result: dict) -> None:
    sep = "=" * 48
    print(f"\n{sep}")
    print("  FOOD DELIVERY PREDICTOR — RESULT")
    print(sep)
    print("\n  Input Summary:")
    for k, v in result["input_summary"].items():
        print(f"    {k:<14}: {v}")
    print()
    print(f"  Predicted Time : {result['predicted_time']} mins")
    print(f"  Status         : {result['status_label']}  "
          f"(Delivery_Status = {result['status_code']})")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    result = predict(
        distance_km = args.distance,
        weather     = args.weather,
        traffic     = args.traffic,
        time_str    = args.time,
    )
    print_result(result)
