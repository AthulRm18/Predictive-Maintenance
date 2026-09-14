"""Feature engineering for C-MAPSS turbofan data.

Computes rolling window statistics, exponentially weighted moving averages,
and sensor deltas to capture degradation trends over engine lifecycle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from machineguard.data.cmapss_loader import get_sensor_columns, get_op_columns


def compute_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling window statistics per engine unit.

    For each sensor and each window size, computes:
    - Rolling mean
    - Rolling standard deviation
    - Rolling min
    - Rolling max

    Args:
        df: C-MAPSS DataFrame with unit_id, cycle, and sensor columns.
        sensor_cols: Sensor columns to compute features for.
        windows: Window sizes in cycles. Default: [5, 10, 20].

    Returns:
        DataFrame with original + rolling feature columns.
    """
    if sensor_cols is None:
        sensor_cols = get_sensor_columns(drop_constant=True)
    if windows is None:
        windows = [5, 10, 20]

    df = df.copy()

    for unit_id in df["unit_id"].unique():
        unit_mask = df["unit_id"] == unit_id

        for col in sensor_cols:
            series = df.loc[unit_mask, col]

            for w in windows:
                rolling = series.rolling(window=w, min_periods=1)
                df.loc[unit_mask, f"{col}_mean_{w}"] = rolling.mean()
                df.loc[unit_mask, f"{col}_std_{w}"] = rolling.std().fillna(0)
                df.loc[unit_mask, f"{col}_min_{w}"] = rolling.min()
                df.loc[unit_mask, f"{col}_max_{w}"] = rolling.max()

    return df


def compute_ewma_features(
    df: pd.DataFrame,
    sensor_cols: list[str] | None = None,
    spans: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Exponentially Weighted Moving Average per engine unit.

    EWMA captures the degradation trend better than simple rolling stats
    because it gives more weight to recent readings.

    Args:
        df: C-MAPSS DataFrame.
        sensor_cols: Sensor columns.
        spans: EWMA span parameters. Default: [5, 10, 20].

    Returns:
        DataFrame with added EWMA columns.
    """
    if sensor_cols is None:
        sensor_cols = get_sensor_columns(drop_constant=True)
    if spans is None:
        spans = [5, 10, 20]

    df = df.copy()

    for unit_id in df["unit_id"].unique():
        unit_mask = df["unit_id"] == unit_id

        for col in sensor_cols:
            series = df.loc[unit_mask, col]

            for span in spans:
                df.loc[unit_mask, f"{col}_ewma_{span}"] = (
                    series.ewm(span=span, min_periods=1).mean()
                )

    return df


def compute_sensor_deltas(
    df: pd.DataFrame,
    sensor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute cycle-over-cycle sensor value changes per engine unit.

    Deltas capture the rate of degradation, which can spike before failure.

    Args:
        df: C-MAPSS DataFrame.
        sensor_cols: Sensor columns.

    Returns:
        DataFrame with added delta columns.
    """
    if sensor_cols is None:
        sensor_cols = get_sensor_columns(drop_constant=True)

    df = df.copy()

    for unit_id in df["unit_id"].unique():
        unit_mask = df["unit_id"] == unit_id

        for col in sensor_cols:
            series = df.loc[unit_mask, col]
            df.loc[unit_mask, f"{col}_delta"] = series.diff().fillna(0)

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    include_rolling: bool = True,
    include_ewma: bool = True,
    include_deltas: bool = True,
    windows: list[int] | None = None,
    ewma_spans: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the full feature matrix for RUL modeling.

    Orchestrates all feature engineering steps and returns a clean
    feature matrix ready for model training.

    Args:
        df: C-MAPSS DataFrame with sensor data and RUL column.
        include_rolling: Whether to add rolling window features.
        include_ewma: Whether to add EWMA features.
        include_deltas: Whether to add sensor delta features.
        windows: Rolling window sizes.
        ewma_spans: EWMA span parameters.

    Returns:
        Tuple of (feature_matrix_df, feature_column_names).
    """
    sensor_cols = get_sensor_columns(drop_constant=True)
    op_cols = get_op_columns()

    df = df.copy()

    # Start with base sensor + operating condition features
    feature_cols = list(sensor_cols) + list(op_cols)

    # Add engineered features
    if include_rolling:
        df = compute_rolling_features(df, sensor_cols, windows)
        w_list = windows or [5, 10, 20]
        for col in sensor_cols:
            for w in w_list:
                feature_cols.extend([
                    f"{col}_mean_{w}", f"{col}_std_{w}",
                    f"{col}_min_{w}", f"{col}_max_{w}",
                ])

    if include_ewma:
        df = compute_ewma_features(df, sensor_cols, ewma_spans)
        span_list = ewma_spans or [5, 10, 20]
        for col in sensor_cols:
            for span in span_list:
                feature_cols.append(f"{col}_ewma_{span}")

    if include_deltas:
        df = compute_sensor_deltas(df, sensor_cols)
        for col in sensor_cols:
            feature_cols.append(f"{col}_delta")

    # Ensure all feature columns exist and fill NaN
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].fillna(0)

    return X, feature_cols
