"""main.py — Unified real-time delivery status prediction API.

Usage (CLI)
-----------
    python main.py --sample sunny_low
    python main.py --json '{"Weather_conditions":"Stormy","Road_traffic_density":"Jam",...}'

Usage (Python import)
---------------------
    from main import predict_delivery_status
    result = predict_delivery_status(my_json_dict)
    print(result["label"], result["probability"])

Model loaded: ANN classifier (models/saved/ann_classifier.keras)
Preprocessing: DataProcessor fitted at startup on the training CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

# Ensure stdout handles UTF-8 on Windows (e.g. PowerShell with cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.processor import DataProcessor
from src.utils.config import (
    ALL_FEATURES, BOOL_FEATURES, CATEGORICAL_FEATURES,
    MODELS_SAVED, NUMERIC_FEATURES, RAW_CSV, RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Model path — ANN was chosen as the deep-learning best model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(MODELS_SAVED, "ann_classifier.keras")

# ---------------------------------------------------------------------------
# Sensible defaults for fields not required in the JSON input
# ---------------------------------------------------------------------------
FIELD_DEFAULTS: dict = {
    "Delivery_person_Age":             30,
    "Delivery_person_Ratings":         4.5,
    "Vehicle_condition":               1,
    "multiple_deliveries":             0.0,
    "Type_of_order":                   "Meal",
    "Type_of_vehicle":                 "motorcycle",
    "City":                            "Urban",
    "Festival":                        "No",
    # Coords default to a ~5 km distance within same city quadrant
    "Restaurant_latitude":             12.9716,
    "Restaurant_longitude":            77.5946,
    "Delivery_location_latitude":      13.0177,
    "Delivery_location_longitude":     77.6408,
    # Order time: mid-morning (non-rush)
    "Time_Orderd":                     "11:00:00",
    # Dummy target — required by DataProcessor.engineer(); not used for inference
    "Time_taken(min)":                 30,
}

VALID_WEATHER = {"Sunny", "Cloudy", "Stormy", "Fog", "Windy", "Sandstorm"}
VALID_TRAFFIC = {"Low", "Medium", "High", "Jam"}

# ---------------------------------------------------------------------------
# Pre-built sample inputs for quick testing
# ---------------------------------------------------------------------------
SAMPLE_INPUTS: dict[str, dict] = {
    "sunny_low": {
        "Weather_conditions":   "Sunny",
        "Road_traffic_density": "Low",
        "Time_Orderd":          "14:00:00",
        "Delivery_person_Age":  28,
        "Restaurant_latitude":  12.9716,
        "Restaurant_longitude": 77.5946,
        "Delivery_location_latitude":  13.0177,
        "Delivery_location_longitude": 77.6408,
    },
    "stormy_jam": {
        "Weather_conditions":   "Stormy",
        "Road_traffic_density": "Jam",
        "Time_Orderd":          "08:30:00",
        "Delivery_person_Age":  35,
        "Restaurant_latitude":  12.9716,
        "Restaurant_longitude": 77.5946,
        "Delivery_location_latitude":  13.1200,
        "Delivery_location_longitude": 77.7200,
        "multiple_deliveries":  1.0,
    },
    "foggy_high_rush": {
        "Weather_conditions":   "Fog",
        "Road_traffic_density": "High",
        "Time_Orderd":          "17:45:00",
        "Restaurant_latitude":  12.9716,
        "Restaurant_longitude": 77.5946,
        "Delivery_location_latitude":  13.0800,
        "Delivery_location_longitude": 77.6900,
    },
}


# ---------------------------------------------------------------------------
# Startup: fit DataProcessor on training data
# ---------------------------------------------------------------------------
_processor: DataProcessor | None = None


def _get_processor() -> DataProcessor:
    """Return a fitted DataProcessor (singleton, fitted once at first call)."""
    global _processor
    if _processor is None:
        if not os.path.exists(RAW_CSV):
            raise FileNotFoundError(
                f"Training CSV not found: {RAW_CSV}\n"
                "Place Food_Delivery_Time_Prediction.csv in data/raw/ first."
            )
        df_train = pd.read_csv(RAW_CSV)
        _processor = DataProcessor()
        _processor.fit(df_train)
    return _processor


# ---------------------------------------------------------------------------
# Startup: load ANN model
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    """Return the loaded Keras ANN (singleton, loaded once at first call)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Saved ANN not found: {MODEL_PATH}\n"
                "Run  python src/models/deep_learning/ann_classifier.py  first."
            )
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------
def predict_delivery_status(input_data: dict) -> dict:
    """Predict delivery status from raw JSON input.

    Parameters
    ----------
    input_data : dict
        Raw field values. Any field not provided will receive a sensible
        default. Required fields for a meaningful prediction:
          - Weather_conditions   (str)  e.g. 'Sunny', 'Stormy'
          - Road_traffic_density (str)  e.g. 'Low', 'High', 'Jam'
          - Time_Orderd          (str)  e.g. '08:30:00'
          - Restaurant_latitude / longitude  (float)
          - Delivery_location_latitude / longitude (float)

    Returns
    -------
    dict with keys:
        label        : str   — 'Fast' or 'Delayed'
        probability  : float — model sigmoid output [0.0, 1.0]
        is_rush_hour : bool  — whether the order time is in a rush window
        distance_km  : float — haversine distance computed from coordinates
        model_used   : str   — name of the loaded model file
    """
    # ── 1. Merge with defaults ────────────────────────────────────
    full_input = {**FIELD_DEFAULTS, **input_data}

    # ── 2. Build a one-row DataFrame with all CSV-expected columns ─
    row_df = pd.DataFrame([full_input])

    # ── 3. Run feature engineering (Haversine, rush-hour, target) ─
    proc = _get_processor()
    row_engineered = proc._engineer(row_df.copy())
    row_imputed    = proc._impute(row_engineered)

    # ── 4. Encode (scale + OHE) ───────────────────────────────────
    X = proc._encode(row_imputed)
    feature_cols = NUMERIC_FEATURES + \
        proc.encoder_.get_feature_names_out(CATEGORICAL_FEATURES).tolist() + \
        BOOL_FEATURES
    X = X[feature_cols].values.astype("float32")

    # ── 5. Infer ──────────────────────────────────────────────────
    model = _get_model()
    probability = float(model.predict(X, verbose=0).ravel()[0])
    label       = "Delayed" if probability >= 0.5 else "Fast"

    return {
        "label":        label,
        "probability":  round(probability, 4),
        "is_rush_hour": bool(row_engineered["is_rush_hour"].iloc[0]),
        "distance_km":  round(float(row_engineered["Distance_km"].iloc[0]), 3),
        "model_used":   os.path.basename(MODEL_PATH),
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def print_result(input_data: dict, result: dict) -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print("  FOOD DELIVERY INTELLIGENCE — PREDICTION")
    print(sep)
    print("\n  Input (key fields):")
    key_fields = ["Weather_conditions", "Road_traffic_density",
                  "Time_Orderd", "Delivery_person_Age", "multiple_deliveries"]
    for k in key_fields:
        v = input_data.get(k, FIELD_DEFAULTS.get(k, "—"))
        print(f"    {k:<32}: {v}")
    print()
    print(f"  Computed — Distance  : {result['distance_km']} km")
    print(f"  Computed — Rush Hour : {result['is_rush_hour']}")
    print()
    prob_pct = result["probability"] * 100
    bar_len  = int(prob_pct / 5)
    bar      = "█" * bar_len + "░" * (20 - bar_len)
    print(f"  Delay Probability : [{bar}] {prob_pct:.1f}%")
    print()
    status_icon = "🔴 DELAYED" if result["label"] == "Delayed" else "🟢 FAST"
    print(f"  ▶  Classification : {status_icon}")
    print(f"     Model          : {result['model_used']}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Food Delivery Intelligence — real-time delivery status predictor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --sample sunny_low\n"
            "  python main.py --sample stormy_jam\n"
            "  python main.py --json '{\"Weather_conditions\":\"Fog\","
            "\"Road_traffic_density\":\"High\",\"Time_Orderd\":\"17:45:00\"}'\n"
            "\nAvailable samples: " + ", ".join(SAMPLE_INPUTS.keys())
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", choices=list(SAMPLE_INPUTS.keys()),
                       help="Run a pre-built sample scenario.")
    group.add_argument("--json", type=str, metavar="JSON_STRING",
                       help="Raw JSON dict of input fields.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    if args.sample:
        input_data = SAMPLE_INPUTS[args.sample]
        print(f"\n  [Using sample: '{args.sample}']")
    else:
        try:
            input_data = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON: {e}")
            sys.exit(1)

    print("  Loading model and processor ...")
    result = predict_delivery_status(input_data)
    print_result(input_data, result)
