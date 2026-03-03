"""Feature engineering and preprocessing pipeline for Food-Delivery-Predictor."""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM     = 6_371.0
MORNING_RUSH        = (7,  9)   # 07:00 – 09:59
EVENING_RUSH        = (17, 20)  # 17:00 – 20:59
DELAY_THRESHOLD_MIN = 40

COL_REST_LAT   = "Restaurant_latitude"
COL_REST_LON   = "Restaurant_longitude"
COL_DEST_LAT   = "Delivery_location_latitude"
COL_DEST_LON   = "Delivery_location_longitude"
COL_TIME_ORDER = "Time_Orderd"   # original column name has a typo – kept as-is
COL_TIME_TAKEN = "Time_taken(min)"


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
class DeliveryPreprocessor:
    """Stateful fit / transform pipeline for delivery-time data.

    Steps (in order)
    ----------------
    1. Impute missing values  – median for numeric, mode for categorical.
    2. Add ``Distance_km``    – great-circle distance via Haversine formula.
    3. Add ``is_rush_hour``   – True if order falls in morning / evening rush.
    4. Add ``Delivery_Status``– 1 (Delayed) if delivery > threshold, else 0.

    Usage
    -----
    >>> pre = DeliveryPreprocessor()
    >>> df_clean = pre.fit_transform(df_raw)
    """

    def __init__(self, delay_threshold: int = DELAY_THRESHOLD_MIN) -> None:
        self.delay_threshold         = delay_threshold
        self.numeric_medians_: dict  = {}
        self.categorical_modes_: dict = {}
        self.is_fitted_              = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DeliveryPreprocessor":
        """Learn imputation statistics from df. Returns self."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols     = df.select_dtypes(include=["object", "category"]).columns

        self.numeric_medians_   = {c: float(df[c].median()) for c in numeric_cols}
        self.categorical_modes_ = {
            c: df[c].mode(dropna=True).iloc[0]
            for c in cat_cols
            if not df[c].mode(dropna=True).empty
        }
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all pipeline steps to df using fitted statistics."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() or fit_transform() before transform().")

        df = df.copy()
        df = self.handle_missing_values(df)
        df = self._add_distance(df)
        df = self._add_is_rush_hour(df)
        df = self._add_delivery_status(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df, then transform and return it."""
        return self.fit(df).transform(df)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute NaNs: median for numeric columns, mode for categorical."""
        df = df.copy()

        for col, median in self.numeric_medians_.items():
            if col in df.columns and df[col].isna().any():
                n = df[col].isna().sum()
                df[col] = df[col].fillna(median)
                print(f"  [impute] {col}: {n} NaN(s) -> median={median:.4f}")

        for col, mode in self.categorical_modes_.items():
            if col in df.columns and df[col].isna().any():
                n = df[col].isna().sum()
                df[col] = df[col].fillna(mode)
                print(f"  [impute] {col}: {n} NaN(s) -> mode='{mode}'")

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine(
        lat1: pd.Series, lon1: pd.Series,
        lat2: pd.Series, lon2: pd.Series,
    ) -> pd.Series:
        """Return vectorised great-circle distance in km (Haversine formula)."""
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)

        a = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        return (EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).round(4)

    def _add_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Distance_km column via Haversine on restaurant / delivery coords."""
        self._require(df, [COL_REST_LAT, COL_REST_LON, COL_DEST_LAT, COL_DEST_LON])
        df["Distance_km"] = self._haversine(
            df[COL_REST_LAT], df[COL_REST_LON],
            df[COL_DEST_LAT], df[COL_DEST_LON],
        )
        return df

    def _add_is_rush_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add is_rush_hour boolean: True during morning (07-09) or evening (17-20) rush."""
        self._require(df, [COL_TIME_ORDER])
        hour = pd.to_datetime(df[COL_TIME_ORDER], format="%H:%M:%S", errors="coerce").dt.hour
        df["is_rush_hour"] = (
            hour.between(*MORNING_RUSH) | hour.between(*EVENING_RUSH)
        ).fillna(False)
        return df

    def _add_delivery_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary Delivery_Status: 1 = Delayed (> threshold), 0 = Fast."""
        self._require(df, [COL_TIME_TAKEN])
        df["Delivery_Status"] = (df[COL_TIME_TAKEN] > self.delay_threshold).astype(int)
        return df

    @staticmethod
    def _require(df: pd.DataFrame, cols: list[str]) -> None:
        """Raise ValueError if any of cols are absent from df."""
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing column(s): {missing}. Got: {df.columns.tolist()}")

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted_ else "not fitted"
        return f"DeliveryPreprocessor(delay_threshold={self.delay_threshold}, {state})"
