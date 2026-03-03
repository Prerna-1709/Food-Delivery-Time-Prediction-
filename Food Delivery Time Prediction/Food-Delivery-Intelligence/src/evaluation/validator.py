"""Validation, cross-validation, and model comparison for Food-Delivery-Intelligence.

Functions
---------
cross_validate_sklearn  -- 5-fold StratifiedKFold CV for any sklearn estimator.
cross_validate_keras    -- 5-fold StratifiedKFold CV for Keras models.
grid_search_ann         -- Manual grid search over ANN (lr × batch_size).
plot_roc_comparison     -- Combined ROC curve for all three models.
run_comparison          -- Orchestrator: trains, validates, plots, and prints table.
"""

from __future__ import annotations

import os
import sys
import itertools
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import (
    RANDOM_SEED, REPORTS_FIGURES, TEST_SIZE, VAL_SIZE,
)

CV_FOLDS   = 5
COLORS     = {"Logistic Regression": "#4C72B0",
              "ANN":                 "#DD8452",
              "CNN (1D)":            "#55A868"}
BG         = "#F7F9FC"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _scalar_metrics(y_true, y_pred, y_prob) -> dict:
    """Return a flat dict of evaluation metrics for one fold."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_prob),
    }


def _mean_std(fold_records: list[dict]) -> dict:
    """Aggregate per-fold metric dicts into mean ± std strings."""
    keys = fold_records[0].keys()
    agg  = {}
    for k in keys:
        vals = [r[k] for r in fold_records]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


# ---------------------------------------------------------------------------
# 1.  Sklearn CV
# ---------------------------------------------------------------------------
def cross_validate_sklearn(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = CV_FOLDS,
    name: str = "Model",
) -> dict:
    """5-fold StratifiedKFold CV for any sklearn estimator with predict_proba."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    fold_records, all_y_true, all_y_prob = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        estimator.fit(X_tr, y_tr)
        y_prob = estimator.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = _scalar_metrics(y_val, y_pred, y_prob)
        fold_records.append(m)
        all_y_true.extend(y_val.tolist())
        all_y_prob.extend(y_prob.tolist())
        print(f"  {name} fold {fold}/{cv} | "
              f"acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  auc={m['auc']:.3f}")

    return {
        "name":       name,
        "stats":      _mean_std(fold_records),
        "y_true_all": np.array(all_y_true),
        "y_prob_all": np.array(all_y_prob),
        "estimator":  estimator,
    }


# ---------------------------------------------------------------------------
# 2.  Keras CV (ANN and CNN share this)
# ---------------------------------------------------------------------------
def cross_validate_keras(
    build_fn,
    X: np.ndarray,
    y: np.ndarray,
    fit_kwargs: dict | None = None,
    cv: int = CV_FOLDS,
    name: str = "Keras Model",
) -> dict:
    """5-fold StratifiedKFold CV for a Keras model factory function.

    Parameters
    ----------
    build_fn  : callable(n_features) -> compiled keras.Model
    X, y      : full numpy arrays (will be split internally)
    fit_kwargs: extra kwargs forwarded to model.fit()
    """
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)

    fit_kwargs = fit_kwargs or {}
    skf        = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    fold_records, all_y_true, all_y_prob = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr_idx].astype("float32"), X[val_idx].astype("float32")
        y_tr, y_val = y[tr_idx].astype("float32"), y[val_idx].astype("float32")
        model = build_fn(X_tr.shape[1])
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  verbose=0, **fit_kwargs)
        y_prob = model.predict(X_val, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        m = _scalar_metrics(y_val.astype(int), y_pred, y_prob)
        fold_records.append(m)
        all_y_true.extend(y_val.astype(int).tolist())
        all_y_prob.extend(y_prob.tolist())
        print(f"  {name} fold {fold}/{cv} | "
              f"acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  auc={m['auc']:.3f}")

    return {
        "name":       name,
        "stats":      _mean_std(fold_records),
        "y_true_all": np.array(all_y_true),
        "y_prob_all": np.array(all_y_prob),
    }


# ---------------------------------------------------------------------------
# 3.  ANN grid search (lr × batch_size)
# ---------------------------------------------------------------------------
def grid_search_ann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    param_grid: dict | None = None,
) -> dict:
    """Manual grid search over ANN learning rate and batch size.

    Parameters
    ----------
    param_grid : dict with keys 'learning_rate' and 'batch_size' (lists).

    Returns
    -------
    dict with keys: best_params, best_val_f1, results_df
    """
    from src.models.deep_learning.ann_classifier import build_model

    param_grid = param_grid or {
        "learning_rate": [1e-2, 1e-3, 5e-4],
        "batch_size":    [8, 16, 32],
    }
    combos   = list(itertools.product(
        param_grid["learning_rate"], param_grid["batch_size"]
    ))
    records  = []
    best_f1  = -1.0
    best_params = {}

    print(f"\n  Grid search over {len(combos)} combinations "
          f"(lr x batch_size) ...")

    for lr, bs in combos:
        model = build_model(X_train.shape[1], learning_rate=lr)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=40, batch_size=bs, verbose=0,
        )
        y_prob = model.predict(X_val, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        f1  = f1_score(y_val.astype(int), y_pred, zero_division=0)
        acc = accuracy_score(y_val.astype(int), y_pred)
        records.append({"lr": lr, "batch_size": bs, "val_f1": round(f1, 4),
                        "val_acc": round(acc, 4)})
        print(f"    lr={lr:.0e}  bs={bs:>2}  |  val_f1={f1:.4f}  val_acc={acc:.4f}")
        if f1 > best_f1:
            best_f1     = f1
            best_params = {"learning_rate": lr, "batch_size": bs}

    results_df = pd.DataFrame(records).sort_values("val_f1", ascending=False)
    print(f"\n  Best params: lr={best_params['learning_rate']:.0e}  "
          f"batch_size={best_params['batch_size']}  val_f1={best_f1:.4f}")
    return {"best_params": best_params, "best_val_f1": best_f1,
            "results_df": results_df}


# ---------------------------------------------------------------------------
# 4.  Combined ROC curve
# ---------------------------------------------------------------------------
def plot_roc_comparison(
    cv_results: list[dict],
    filename: str = "roc_comparison.png",
) -> str:
    """Plot combined ROC curves from cross-validation OOF probabilities."""
    os.makedirs(REPORTS_FIGURES, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for res in cv_results:
        name   = res["name"]
        color  = COLORS.get(name, "#8C8C8C")
        y_true = res["y_true_all"]
        y_prob = res["y_prob_all"]

        # Guard: skip if only one class present in OOF predictions
        if len(np.unique(y_true)) < 2:
            print(f"  [warn] {name}: only one class in OOF - skipping ROC")
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title("ROC Comparison — 5-Fold OOF Scores",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()

    path = os.path.join(REPORTS_FIGURES, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ---------------------------------------------------------------------------
# 5.  Comparison table
# ---------------------------------------------------------------------------
def print_comparison_table(cv_results: list[dict]) -> pd.DataFrame:
    """Print a final ranked comparison table (sorted by F1-score)."""
    rows = []
    for res in cv_results:
        s = res["stats"]
        rows.append({
            "Model":     res["name"],
            "Accuracy":  f"{s['accuracy']['mean']:.4f} ± {s['accuracy']['std']:.4f}",
            "Precision": f"{s['precision']['mean']:.4f} ± {s['precision']['std']:.4f}",
            "Recall":    f"{s['recall']['mean']:.4f} ± {s['recall']['std']:.4f}",
            "F1-Score":  f"{s['f1']['mean']:.4f} ± {s['f1']['std']:.4f}",
            "AUC-ROC":   f"{s['auc']['mean']:.4f} ± {s['auc']['std']:.4f}",
            "_f1_mean":  s["f1"]["mean"],
        })

    df = (pd.DataFrame(rows)
            .sort_values("_f1_mean", ascending=False)
            .drop(columns="_f1_mean")
            .reset_index(drop=True))

    sep = "=" * 95
    print(f"\n{sep}")
    print("  FINAL MODEL COMPARISON  (5-Fold CV, sorted by F1-Score)")
    print(sep)
    header = (f"  {'Model':<24} {'Accuracy':>18} {'Precision':>18}"
              f" {'Recall':>18} {'F1-Score':>18} {'AUC-ROC':>18}")
    print(header)
    print("  " + "-" * 91)
    for _, row in df.iterrows():
        winner = "  <-- BEST" if _ == 0 else ""
        print(f"  {row['Model']:<24} {row['Accuracy']:>18} {row['Precision']:>18}"
              f" {row['Recall']:>18} {row['F1-Score']:>18} {row['AUC-ROC']:>18}{winner}")
    print(sep + "\n")
    return df


# ---------------------------------------------------------------------------
# 6.  Orchestrator
# ---------------------------------------------------------------------------
def run_comparison(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full validation pipeline: CV → ROC → grid search → comparison table."""
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)

    from src.preprocessing.processor import DataProcessor
    from src.models.deep_learning.ann_classifier import build_model as build_ann

    sep = "=" * 60

    # ── Data prep ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STEP 6 -- CROSS-VALIDATION & MODEL COMPARISON")
    print(sep)

    proc = DataProcessor()
    X_tr_full, X_test, y_tr_full, y_test = proc.fit_transform(df_raw)

    X_np  = np.vstack([X_tr_full.values, X_test.values]).astype("float32")
    y_np  = np.concatenate([y_tr_full.values, y_test.values]).astype(int)
    n_feats = X_np.shape[1]
    print(f"\n  Dataset: {X_np.shape[0]} rows, {n_feats} features  |  "
          f"Delayed={y_np.sum()}, Fast={(y_np==0).sum()}")

    # ── 1. Logistic Regression CV ──────────────────────────────────
    print(f"\n[1] Logistic Regression — {CV_FOLDS}-fold StratifiedKFold ...")
    lr_model = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
    lr_res   = cross_validate_sklearn(lr_model, X_np, y_np, name="Logistic Regression")

    # ── 2. ANN CV ─────────────────────────────────────────────────
    print(f"\n[2] ANN — {CV_FOLDS}-fold StratifiedKFold ...")
    ann_res = cross_validate_keras(
        build_fn=build_ann,
        X=X_np, y=y_np.astype("float32"),
        fit_kwargs={"epochs": 50, "batch_size": 16},
        name="ANN",
    )

    # ── 3. 1D-CNN CV ─────────────────────────────────────────────
    print(f"\n[3] CNN (1D) — {CV_FOLDS}-fold StratifiedKFold ...")

    def build_cnn(n_features: int) -> tf.keras.Model:
        """Lightweight 1D-CNN for tabular data (reshape to [batch, features, 1])."""
        inp = tf.keras.Input(shape=(n_features,), name="input")
        x   = tf.keras.layers.Reshape((n_features, 1))(inp)
        x   = tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu",
                                     padding="same")(x)
        x   = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x   = tf.keras.layers.Conv1D(16, kernel_size=3, activation="relu",
                                     padding="same")(x)
        x   = tf.keras.layers.GlobalAveragePooling1D()(x)
        x   = tf.keras.layers.Dropout(0.2)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inp, out, name="cnn_1d")
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    cnn_res = cross_validate_keras(
        build_fn=build_cnn,
        X=X_np, y=y_np.astype("float32"),
        fit_kwargs={"epochs": 50, "batch_size": 16},
        name="CNN (1D)",
    )

    # ── 4. Combined ROC curve ─────────────────────────────────────
    print(f"\n[4] Plotting combined ROC curves ...")
    cv_results = [lr_res, ann_res, cnn_res]
    roc_path   = plot_roc_comparison(cv_results)

    # ── 5. ANN grid search ────────────────────────────────────────
    print(f"\n[5] ANN Hyperparameter Grid Search ...")
    # Use 80% of data for train and 20% for validation
    X_gs_tr, X_gs_val, y_gs_tr, y_gs_val = train_test_split(
        X_np, y_np.astype("float32"),
        test_size=0.20, random_state=RANDOM_SEED, stratify=y_np,
    )
    gs_result = grid_search_ann(X_gs_tr, y_gs_tr, X_gs_val, y_gs_val)
    print(f"\n  Grid Search Results (all combos):")
    print("  " + gs_result["results_df"].to_string(index=False))

    # ── 6. Final comparison table ─────────────────────────────────
    print(f"\n[6] Final Comparison Table:")
    comparison_df = print_comparison_table(cv_results)

    return comparison_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_raw = pd.read_csv(
        os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))),
            "data", "raw", "Food_Delivery_Time_Prediction.csv")
    )
    run_comparison(df_raw)
