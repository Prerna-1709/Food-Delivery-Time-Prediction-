"""ANN binary classifier for Delivery_Status prediction (Fast=0 / Delayed=1).

Architecture
------------
Input  →  Dense(64, relu) → Dropout(0.2)
       →  Dense(32, relu) → Dropout(0.2)
       →  Dense(1,  sigmoid)

Training
--------
Optimizer : Adam
Loss      : binary_crossentropy
Metrics   : accuracy, AUC
Callbacks : EarlyStopping on val_loss (patience=DL_PATIENCE)
            ModelCheckpoint (best weights saved automatically)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info / warning logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.utils.config import (
    DL_BATCH_SIZE, DL_EPOCHS, DL_PATIENCE,
    MODELS_SAVED, RANDOM_SEED,
)

# Reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SAVE_PATH = os.path.join(MODELS_SAVED, "ann_classifier.keras")


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(n_features: int, learning_rate: float = 1e-3) -> keras.Model:
    """Return a compiled Keras Sequential ANN.

    Parameters
    ----------
    n_features    : int   — number of input features (after encoding).
    learning_rate : float — Adam learning rate.

    Returns
    -------
    keras.Model (compiled, not yet trained)
    """
    model = keras.Sequential([
        keras.Input(shape=(n_features,), name="input"),

        layers.Dense(64, activation="relu", name="hidden_1"),
        layers.Dropout(0.2, name="dropout_1"),

        layers.Dense(32, activation="relu", name="hidden_2"),
        layers.Dropout(0.2, name="dropout_2"),

        layers.Dense(1, activation="sigmoid", name="output"),
    ], name="ann_delivery_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    epochs:     int = DL_EPOCHS,
    batch_size: int = DL_BATCH_SIZE,
    patience:   int = DL_PATIENCE,
    save_path:  str = SAVE_PATH,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train the ANN and return the best model + training history.

    Early stopping monitors val_loss; best weights are restored automatically.
    The best model is saved to *save_path* in native Keras format.

    Parameters
    ----------
    X_train, y_train : training data (numpy arrays).
    X_val,   y_val   : validation data.
    epochs           : maximum training epochs.
    batch_size       : mini-batch size.
    patience         : EarlyStopping patience (epochs without val_loss improvement).
    save_path        : where to persist the best model.

    Returns
    -------
    model   : keras.Model — best weights restored.
    history : keras.callbacks.History — epoch-by-epoch metrics.
    """
    n_features = X_train.shape[1]
    model = build_model(n_features)

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1,
    )
    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Run full evaluation on the test set and return metrics dict.

    Returns accuracy, AUC, precision, recall, F1, and a confusion matrix.
    """
    from sklearn.metrics import (
        accuracy_score, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)),            4),
        "auc":       round(float(roc_auc_score(y_test, y_prob)),              4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)),    4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)),        4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def print_history(history: keras.callbacks.History) -> None:
    """Print a per-epoch loss / accuracy table."""
    hist = history.history
    epochs_run = len(hist["loss"])
    sep = "-" * 70
    header = f"  {'Epoch':>5}  {'loss':>8}  {'acc':>8}  {'auc':>8}  {'val_loss':>10}  {'val_acc':>9}  {'val_auc':>9}"
    print(f"\n  Training History ({epochs_run} epochs):")
    print("  " + sep)
    print(header)
    print("  " + sep)
    for i in range(epochs_run):
        print(
            f"  {i+1:>5}  "
            f"{hist['loss'][i]:>8.4f}  "
            f"{hist['accuracy'][i]:>8.4f}  "
            f"{hist['auc'][i]:>8.4f}  "
            f"{hist['val_loss'][i]:>10.4f}  "
            f"{hist['val_accuracy'][i]:>9.4f}  "
            f"{hist['val_auc'][i]:>9.4f}"
        )
    print("  " + sep)


def print_metrics(metrics: dict) -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print("  ANN CLASSIFIER — TEST SET METRICS")
    print(sep)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  AUC       : {metrics['auc']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"             Pred Fast  Pred Delayed")
    print(f"  True Fast      {cm[0][0]:>4}         {cm[0][1]:>4}")
    print(f"  True Delayed   {cm[1][0]:>4}         {cm[1][1]:>4}")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from src.preprocessing.processor import DataProcessor
    from src.utils.config import RAW_CSV, TEST_SIZE, VAL_SIZE

    sep = "=" * 52
    print(f"\n{sep}")
    print("  STEP 4 -- ANN BINARY CLASSIFIER")
    print(sep)

    # ── Data prep ─────────────────────────────────────────────────
    print("\n[1] Loading and preprocessing ...")
    df_raw = pd.read_csv(RAW_CSV)
    proc   = DataProcessor()
    X_train_full, X_test, y_train_full, y_test = proc.fit_transform(df_raw)

    # Carve out a validation split from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED, stratify=y_train_full,
    )
    print(f"    Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"    Features: {X_train.shape[1]}")

    # Convert to numpy
    X_train_np = X_train.values.astype("float32")
    X_val_np   = X_val.values.astype("float32")
    X_test_np  = X_test.values.astype("float32")
    y_train_np = y_train.values.astype("float32")
    y_val_np   = y_val.values.astype("float32")
    y_test_np  = y_test.values.astype("float32")

    # ── Model summary ─────────────────────────────────────────────
    print("\n[2] Model Architecture:")
    model = build_model(X_train_np.shape[1])
    model.summary()

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n[3] Training (max {DL_EPOCHS} epochs, early stopping patience={DL_PATIENCE}) ...")
    model, history = train(
        X_train_np, y_train_np,
        X_val_np,   y_val_np,
    )

    # ── History table ─────────────────────────────────────────────
    print_history(history)

    # ── Evaluate ──────────────────────────────────────────────────
    print("\n[4] Evaluating on test set ...")
    metrics = evaluate(model, X_test_np, y_test_np)
    print_metrics(metrics)

    # ── Confirm loss trend ────────────────────────────────────────
    losses = history.history["loss"]
    is_decreasing = losses[-1] < losses[0]
    print(f"\n[5] Loss trend:  {losses[0]:.4f} -> {losses[-1]:.4f}  "
          f"({'DECREASING' if is_decreasing else 'NOT DECREASING'})")

    # ── Save ──────────────────────────────────────────────────────
    size_kb = os.path.getsize(SAVE_PATH) / 1024 if os.path.exists(SAVE_PATH) else 0
    print(f"\n[6] Model saved: {SAVE_PATH}  ({size_kb:.1f} KB)")
    print(f"\n[DONE]\n")
