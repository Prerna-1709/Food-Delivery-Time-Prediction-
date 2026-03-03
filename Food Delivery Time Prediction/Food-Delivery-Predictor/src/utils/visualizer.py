"""Visualization utilities for the Food-Delivery-Predictor project.

Functions
---------
plot_confusion_matrix  -- Heatmap of a classifier's confusion matrix.
plot_roc_curve         -- ROC curve + AUC for one or more classifiers.
plot_correlation_heatmap -- Pearson correlation heatmap of numeric features.
generate_all_reports   -- Orchestrates all plots and saves them to reports/figures/.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts with no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
PALETTE    = "Blues"
ACCENT     = "#4C72B0"
BACKGROUND = "#F7F9FC"
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor": BACKGROUND,
    "axes.facecolor":   BACKGROUND,
})


def _savefig(fig: plt.Figure, filename: str) -> str:
    """Save *fig* to FIGURES_DIR/<filename> and return the full path."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Confusion Matrix Heatmap
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "Model",
    class_labels: list[str] | None = None,
    filename: str | None = None,
) -> str:
    """Plot and save a styled confusion-matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted labels.
    model_name : str
        Title prefix (e.g. ``"Naive Bayes"``).
    class_labels : list[str], optional
        Display labels for the classes. Defaults to ``["Fast", "Delayed"]``.
    filename : str, optional
        Output filename. Auto-generated from *model_name* if omitted.

    Returns
    -------
    str – absolute path to the saved PNG.
    """
    class_labels = class_labels or ["Fast (0)", "Delayed (1)"]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=PALETTE,
        xticklabels=class_labels, yticklabels=class_labels,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()

    fname = filename or f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    return _savefig(fig, fname)


# ---------------------------------------------------------------------------
# 2. ROC Curve + AUC
# ---------------------------------------------------------------------------
def plot_roc_curve(
    models: list[tuple[str, object]],
    X_test: pd.DataFrame,
    y_test,
    filename: str = "roc_curves.png",
) -> str:
    """Plot ROC curves and AUC for one or more fitted classifiers.

    Parameters
    ----------
    models : list of (name, fitted_pipeline) tuples
        Each pipeline must expose ``predict_proba``.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : array-like
        True binary labels.
    filename : str
        Output filename inside FIGURES_DIR.

    Returns
    -------
    str – absolute path to the saved PNG.
    """
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (name, pipeline) in enumerate(models):
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        except AttributeError:
            print(f"  [warn] {name} does not support predict_proba – skipping ROC.")
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                label=f"{name}  (AUC = {auc:.3f})")

    # Diagonal baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")

    ax.set_title("ROC Curves — Classifier Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()

    return _savefig(fig, filename)


# ---------------------------------------------------------------------------
# 3. Correlation Heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(
    df: pd.DataFrame,
    filename: str = "correlation_heatmap.png",
) -> str:
    """Plot and save a Pearson correlation heatmap for numeric columns in df.

    Only numeric and boolean columns are included. The target column
    ``Time_taken(min)`` is highlighted via column ordering (the first column)
    so you can quickly read which features correlate most with delivery time.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe.
    filename : str
        Output filename inside FIGURES_DIR.

    Returns
    -------
    str – absolute path to the saved PNG.
    """
    # Select numeric + boolean columns; cast booleans to int
    num_df = df.select_dtypes(include=[np.number, bool]).copy()
    for col in num_df.select_dtypes(include=bool).columns:
        num_df[col] = num_df[col].astype(int)

    # Drop identifier columns that carry no analytical meaning
    drop_cols = [c for c in ["ID"] if c in num_df.columns]
    num_df = num_df.drop(columns=drop_cols)

    # Reorder so delivery time is first — makes row/column reading intuitive
    target = "Time_taken(min)"
    if target in num_df.columns:
        cols = [target] + [c for c in num_df.columns if c != target]
        num_df = num_df[cols]

    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle masked
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="white",
        annot_kws={"size": 9},
        square=True, ax=ax,
    )
    ax.set_title(
        "Feature Correlation Heatmap\n(Pearson — lower triangle)",
        fontsize=13, fontweight="bold", pad=16,
    )
    ax.tick_params(axis="both", labelsize=9, rotation=45)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    return _savefig(fig, filename)


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------
def generate_all_reports(df: pd.DataFrame) -> None:
    """Train classifiers internally and generate all visual reports.

    Saves the following PNGs to ``reports/figures/``:
      - confusion_matrix_naive_bayes.png
      - confusion_matrix_knn.png
      - confusion_matrix_decision_tree.png
      - roc_curves.png
      - correlation_heatmap.png

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe (output of DeliveryPreprocessor).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from src.models.classifier_suite import (
        BOOL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
        build_decision_tree, build_knn, build_naive_bayes,
    )

    print("\n  Preparing data ...")
    le  = LabelEncoder()
    y   = le.fit_transform(df["Delivery_Status"])
    X   = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOL_FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Train all three models
    print("  Training models ...")
    nb = build_naive_bayes();              nb.fit(X_train, y_train)
    knn, best_k = build_knn(X_train, y_train)
    dt,  best_d = build_decision_tree(X_train, y_train)

    model_map = {
        "Naive Bayes":              nb,
        f"KNN (K={best_k})":       knn,
        f"Decision Tree (d={best_d})": dt,
    }

    # ── Confusion matrices ────────────────────────────────────────────────────
    print("\n  Generating confusion matrices ...")
    for name, pipeline in model_map.items():
        y_pred = pipeline.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, model_name=name)

    # ── ROC curves ────────────────────────────────────────────────────────────
    print("  Generating ROC curves ...")
    plot_roc_curve(list(model_map.items()), X_test, y_test)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    print("  Generating correlation heatmap ...")
    plot_correlation_heatmap(df)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.preprocessing.engineer import DeliveryPreprocessor
    from src.utils.data_loader import load_data

    sep = "=" * 55
    print(f"\n{sep}")
    print("  STEP 5 -- VISUALIZATION & REPORTING")
    print(sep)

    print("\n[1] Loading and engineering features ...")
    df = DeliveryPreprocessor().fit_transform(load_data())
    print(f"    Shape: {df.shape}")

    print("\n[2] Generating all figures ...")
    generate_all_reports(df)

    print(f"\n[3] Checking reports/figures/ ...")
    for f in sorted(os.listdir(FIGURES_DIR)):
        size = os.path.getsize(os.path.join(FIGURES_DIR, f))
        print(f"    {f:<50} ({size/1024:.1f} KB)")

    print(f"\n[DONE] All figures saved to:  {FIGURES_DIR}\n")
