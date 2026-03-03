"""DataProcessor: preprocessing + feature engineering for Food-Delivery-Intelligence.

Pipeline stages (in order)
--------------------------
1. Haversine distance from lat/lon coordinates.
2. Rush-hour boolean from order timestamp.
3. Binary Delivery_Status target (0 = Fast, 1 = Delayed).
4. Missing-value imputation (median numeric / mode categorical).
5. StandardScaler on continuous features; OneHotEncoder on categoricals.

Usage
-----
>>> proc = DataProcessor()
>>> X_train_enc, X_test_enc, y_train, y_test = proc.fit_transform(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import (
    ALL_FEATURES, BOOL_FEATURES, CATEGORICAL_FEATURES,
    DELAY_THRESHOLD_MIN, NUMERIC_FEATURES, RANDOM_SEED, TEST_SIZE,
)

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
COL_REST_LAT   = "Restaurant_latitude"
COL_REST_LON   = "Restaurant_longitude"
COL_DEST_LAT   = "Delivery_location_latitude"
COL_DEST_LON   = "Delivery_location_longitude"
COL_TIME_ORDER = "Time_Orderd"        # original CSV typo kept intentional
COL_TIME_TAKEN = "Time_taken(min)"
COL_TARGET     = "Delivery_Status"

EARTH_RADIUS_KM = 6_371.0
MORNING_RUSH    = (7,  9)    # 07:00 – 09:59
EVENING_RUSH    = (17, 20)   # 17:00 – 20:59


# ---------------------------------------------------------------------------
# DataProcessor
# ---------------------------------------------------------------------------
class DataProcessor:
    """Stateful preprocessing + encoding pipeline.

    After calling fit(), the fitted StandardScaler and OneHotEncoder are
    stored as attributes so they can be reused on test / inference data.

    Attributes
    ----------
    scaler_        : StandardScaler — fitted on numeric training features.
    encoder_       : OneHotEncoder  — fitted on categorical training features.
    feature_names_ : list[str]      — final encoded column names (set after fit).
    numeric_medians_   : dict       — fitted medians for imputation.
    categorical_modes_ : dict       — fitted modes for imputation.
    is_fitted_     : bool
    """

    def __init__(self, delay_threshold: int = DELAY_THRESHOLD_MIN) -> None:
        self.delay_threshold       = delay_threshold
        self.scaler_               = StandardScaler()
        self.encoder_              = OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False)
        self.numeric_medians_: dict  = {}
        self.categorical_modes_: dict = {}
        self.feature_names_: list[str] = []
        self.is_fitted_            = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DataProcessor":
        """Learn statistics (medians, modes, scaler, encoder) from df."""
        df = self._engineer(df.copy())

        # Imputation stats
        num_cols = df[NUMERIC_FEATURES].select_dtypes(include=[np.number]).columns
        self.numeric_medians_   = {c: float(df[c].median()) for c in num_cols}
        self.categorical_modes_ = {
            c: df[c].mode(dropna=True).iloc[0]
            for c in CATEGORICAL_FEATURES
            if not df[c].mode(dropna=True).empty
        }

        df = self._impute(df)

        # Fit scaler & encoder
        self.scaler_.fit(df[NUMERIC_FEATURES])
        self.encoder_.fit(df[CATEGORICAL_FEATURES])

        # Cache final feature names
        ohe_cols = self.encoder_.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        self.feature_names_ = NUMERIC_FEATURES + ohe_cols + BOOL_FEATURES
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Apply the full pipeline and return (X_encoded, y).

        Returns
        -------
        X : pd.DataFrame  — full encoded feature matrix, no nulls.
        y : pd.Series     — Delivery_Status target (0/1).
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() or fit_transform() before transform().")

        df = self._engineer(df.copy())
        df = self._impute(df)
        y  = df[COL_TARGET]
        X  = self._encode(df)
        return X, y

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Fit, engineer, encode, split into train / test sets.

        Returns
        -------
        X_train, X_test, y_train, y_test  — all as pandas objects.
        """
        self.fit(df)
        X, y = self.transform(df)
        return train_test_split(X, y, test_size=TEST_SIZE,
                                random_state=RANDOM_SEED, stratify=y)

    # ------------------------------------------------------------------
    # Stage 1 — Feature Engineering
    # ------------------------------------------------------------------

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Haversine, rush-hour, and target-generation in order."""
        df = self._add_distance(df)
        df = self._add_rush_hour(df)
        df = self._add_target(df)
        return df

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> pd.Series:
        """Vectorised great-circle distance in km."""
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)
        a = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        return (EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).round(4)

    def _add_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Distance_km via Haversine on restaurant / delivery coordinates."""
        self._require(df, [COL_REST_LAT, COL_REST_LON, COL_DEST_LAT, COL_DEST_LON])
        df["Distance_km"] = self._haversine(
            df[COL_REST_LAT], df[COL_REST_LON],
            df[COL_DEST_LAT], df[COL_DEST_LON],
        )
        return df

    @staticmethod
    def _add_rush_hour(df: pd.DataFrame) -> pd.DataFrame:
        """Add is_rush_hour boolean: True during morning (07-09) or evening (17-20)."""
        hour = pd.to_datetime(df[COL_TIME_ORDER], format="%H:%M:%S", errors="coerce").dt.hour
        df["is_rush_hour"] = (
            hour.between(*MORNING_RUSH) | hour.between(*EVENING_RUSH)
        ).fillna(False)
        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary Delivery_Status: 1 = Delayed (> threshold), 0 = Fast."""
        self._require(df, [COL_TIME_TAKEN])
        df[COL_TARGET] = (df[COL_TIME_TAKEN] > self.delay_threshold).astype(int)
        return df

    # ------------------------------------------------------------------
    # Stage 2 — Imputation
    # ------------------------------------------------------------------

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs using fitted medians (numeric) and modes (categorical)."""
        df = df.copy()
        for col, median in self.numeric_medians_.items():
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing:
                    df[col] = df[col].fillna(median)
                    print(f"  [impute] {col}: {missing} NaN(s) -> median={median:.4f}")
        for col, mode in self.categorical_modes_.items():
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing:
                    df[col] = df[col].fillna(mode)
                    print(f"  [impute] {col}: {missing} NaN(s) -> mode='{mode}'")
        return df

    # ------------------------------------------------------------------
    # Stage 3 — Encoding (StandardScaler + OneHotEncoder)
    # ------------------------------------------------------------------

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features and one-hot encode categoricals.

        Returns a single DataFrame with columns:
            [scaled numeric cols] + [OHE columns] + [bool cols]
        """
        # Numeric → StandardScaler
        num_scaled = self.scaler_.transform(df[NUMERIC_FEATURES])
        num_df     = pd.DataFrame(num_scaled, columns=NUMERIC_FEATURES,
                                  index=df.index)

        # Categorical → OneHotEncoder
        cat_enc  = self.encoder_.transform(df[CATEGORICAL_FEATURES])
        cat_cols = self.encoder_.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        cat_df   = pd.DataFrame(cat_enc, columns=cat_cols, index=df.index)

        # Boolean → passthrough as int
        bool_df  = df[BOOL_FEATURES].astype(int).reset_index(drop=True)
        num_df   = num_df.reset_index(drop=True)
        cat_df   = cat_df.reset_index(drop=True)

        return pd.concat([num_df, cat_df, bool_df], axis=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require(df: pd.DataFrame, cols: list[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted_ else "not fitted"
        n_features = len(self.feature_names_)
        return (f"DataProcessor(delay_threshold={self.delay_threshold}, "
                f"{state}, output_features={n_features})")
