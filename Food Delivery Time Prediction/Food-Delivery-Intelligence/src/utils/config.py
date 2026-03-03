"""Global configuration constants for Food-Delivery-Intelligence."""

from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Project root — resolves correctly regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_RAW_DIR    = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC_DIR   = os.path.join(PROJECT_ROOT, "data", "processed")

MODELS_SAVED    = os.path.join(PROJECT_ROOT, "models", "saved")
REPORTS_FIGURES = os.path.join(PROJECT_ROOT, "reports", "figures")
NOTEBOOKS_DIR   = os.path.join(PROJECT_ROOT, "notebooks")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
RAW_CSV = os.path.join(DATA_RAW_DIR, "Food_Delivery_Time_Prediction.csv")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
VAL_SIZE  = 0.10   # fraction of training set used for validation (deep learning)

# ---------------------------------------------------------------------------
# Model targets
# ---------------------------------------------------------------------------
TARGET_REGRESSION     = "Time_taken(min)"      # continuous output
TARGET_CLASSIFICATION = "Delivery_Status"      # binary 0/1

# ---------------------------------------------------------------------------
# Delay threshold (minutes) — used by DeliveryPreprocessor
# ---------------------------------------------------------------------------
DELAY_THRESHOLD_MIN = 40

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Distance_km",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
]

CATEGORICAL_FEATURES = [
    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "City",
    "Festival",
]

BOOL_FEATURES = ["is_rush_hour"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOL_FEATURES

# ---------------------------------------------------------------------------
# Deep learning hyperparameters (defaults — override in model scripts)
# ---------------------------------------------------------------------------
DL_EPOCHS     = 100
DL_BATCH_SIZE = 32
DL_PATIENCE   = 10     # early-stopping patience

# ---------------------------------------------------------------------------
# Ensure output directories exist on import
# ---------------------------------------------------------------------------
for _dir in [DATA_RAW_DIR, DATA_PROC_DIR, MODELS_SAVED, REPORTS_FIGURES]:
    os.makedirs(_dir, exist_ok=True)
