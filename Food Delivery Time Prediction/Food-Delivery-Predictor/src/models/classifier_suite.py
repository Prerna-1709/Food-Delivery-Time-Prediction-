"""Classification suite: Naive Bayes, KNN, Decision Tree for Delivery_Status prediction."""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.engineer import DeliveryPreprocessor  # noqa: E402
from src.utils.data_loader import load_data                  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_COL  = "Delivery_Status"   # binary: 0 = Fast, 1 = Delayed
TEST_SIZE   = 0.20
RANDOM_SEED = 42
SAVE_DIR    = os.path.join(PROJECT_ROOT, "models", "saved")

NUMERIC_FEATURES     = ["Distance_km", "Delivery_person_Age", "Delivery_person_Ratings",
                         "Vehicle_condition", "multiple_deliveries"]
CATEGORICAL_FEATURES = ["Weather_conditions", "Road_traffic_density",
                         "Type_of_order", "Type_of_vehicle", "City", "Festival"]
BOOL_FEATURES        = ["is_rush_hour"]

# KNN: candidate K values for grid search
KNN_K_GRID = {"classifier__n_neighbors": list(range(1, 12, 2))}

# Decision Tree: candidate max_depth values for pruning
DT_DEPTH_GRID = {"classifier__max_depth": [2, 3, 4, 5, None]}


# ---------------------------------------------------------------------------
# Preprocessing transformer (shared across all classifiers)
# ---------------------------------------------------------------------------
def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num",  StandardScaler(),                                         NUMERIC_FEATURES),
            ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("bool", "passthrough",                                            BOOL_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_naive_bayes() -> Pipeline:
    """Gaussian Naive Bayes pipeline."""
    return Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("classifier",   GaussianNB()),
    ])


def build_knn(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, int]:
    """KNN pipeline with GridSearchCV to find the optimal K.

    Returns the best fitted pipeline and the best K value.
    """
    base = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("classifier",   KNeighborsClassifier()),
    ])
    search = GridSearchCV(
        base, KNN_K_GRID,
        cv=min(5, len(X_train)),   # guard against small datasets
        scoring="f1_weighted",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_k = search.best_params_["classifier__n_neighbors"]
    return search.best_estimator_, best_k


def build_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, int | None]:
    """Decision Tree with GridSearchCV over max_depth (pruning).

    Returns the best fitted pipeline and the best max_depth.
    """
    base = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("classifier",   DecisionTreeClassifier(random_state=RANDOM_SEED)),
    ])
    search = GridSearchCV(
        base, DT_DEPTH_GRID,
        cv=min(5, len(X_train)),
        scoring="f1_weighted",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_depth = search.best_params_["classifier__max_depth"]
    return search.best_estimator_, best_depth


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _metrics(name: str, y_true, y_pred) -> dict:
    """Compute accuracy, precision, recall, and F1 for one model."""
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred),               4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0),    4),
        "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0),        4),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Return a formatted comparison DataFrame sorted by F1-Score."""
    df = pd.DataFrame(results).set_index("Model")
    return df.sort_values("F1-Score", ascending=False)


def print_comparison(df: pd.DataFrame) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  MODEL COMPARISON  (sorted by F1-Score, descending)")
    print(sep)
    # Header
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>10}")
    print("  " + "-" * 61)
    for model, row in df.iterrows():
        winner = " <-- BEST" if model == df.index[0] else ""
        print(
            f"  {model:<22} {row['Accuracy']:>9.4f} {row['Precision']:>10.4f}"
            f" {row['Recall']:>8.4f} {row['F1-Score']:>10.4f}{winner}"
        )
    print(sep)


def print_report(name: str, y_true, y_pred) -> None:
    print(f"\n  [{name}] Classification Report:")
    print("  " + "-" * 40)
    report = classification_report(
        y_true, y_pred,
        target_names=["Fast (0)", "Delayed (1)"],
        zero_division=0,
    )
    for line in report.splitlines():
        print("  " + line)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run() -> pd.DataFrame:
    """Load data, train all three classifiers, compare, and save models."""
    sep = "=" * 65

    print(f"\n{sep}")
    print("  STEP 4 -- CLASSIFICATION SUITE")
    print(sep)

    # ── 1. Data prep ─────────────────────────────────────────────────────────
    print("\n[1] Loading and engineering features ...")
    raw_df = load_data()
    preprocessor = DeliveryPreprocessor()
    df = preprocessor.fit_transform(raw_df)
    print(f"    Shape: {df.shape}")

    # LabelEncoder on the binary target (already 0/1 int, but kept for explicit compliance)
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"    Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    print(f"    Target classes: {le.classes_.tolist()} (encoded as {list(range(len(le.classes_)))})")

    results = []

    # ── 2. Gaussian Naive Bayes ───────────────────────────────────────────────
    print("\n[2] Training Gaussian Naive Bayes ...")
    nb_pipeline = build_naive_bayes()
    nb_pipeline.fit(X_train, y_train)
    y_pred_nb = nb_pipeline.predict(X_test)
    print_report("Naive Bayes", y_test, y_pred_nb)
    results.append(_metrics("Naive Bayes", y_test, y_pred_nb))

    # ── 3. KNN + GridSearchCV ─────────────────────────────────────────────────
    print("\n[3] Training KNN (GridSearchCV over K) ...")
    knn_pipeline, best_k = build_knn(X_train, y_train)
    print(f"    Best K found by GridSearchCV: {best_k}")
    y_pred_knn = knn_pipeline.predict(X_test)
    print_report(f"KNN (K={best_k})", y_test, y_pred_knn)
    results.append(_metrics(f"KNN (K={best_k})", y_test, y_pred_knn))

    # ── 4. Decision Tree + pruning ────────────────────────────────────────────
    print("\n[4] Training Decision Tree (GridSearchCV over max_depth) ...")
    dt_pipeline, best_depth = build_decision_tree(X_train, y_train)
    depth_str = str(best_depth) if best_depth else "None (unpruned)"
    print(f"    Best max_depth: {depth_str}")
    y_pred_dt = dt_pipeline.predict(X_test)
    print_report(f"Decision Tree (depth={best_depth})", y_test, y_pred_dt)
    results.append(_metrics(f"Decision Tree (d={best_depth})", y_test, y_pred_dt))

    # ── 5. Comparison table ───────────────────────────────────────────────────
    comparison_df = compare_models(results)
    print_comparison(comparison_df)

    # ── 6. Save all models ────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("\n[5] Saving models ...")
    for pipeline, fname in [
        (nb_pipeline,  "naive_bayes.joblib"),
        (knn_pipeline, "knn.joblib"),
        (dt_pipeline,  "decision_tree.joblib"),
    ]:
        path = os.path.join(SAVE_DIR, fname)
        joblib.dump(pipeline, path)
        print(f"    Saved: {path}  ({os.path.getsize(path)/1024:.1f} KB)")

    print(f"\n[DONE] Classification suite complete.\n")
    return comparison_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
