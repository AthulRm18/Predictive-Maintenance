"""Feature engineering for AI4I 2020 Predictive Maintenance data.

Physics-based derived features that mirror actual failure mode thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from machineguard.data.ai4i_loader import SENSOR_COLUMNS, SENSOR_NAME_MAP


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-based features that correspond to failure mode logic.

    These features are derived from the dataset documentation:
    - PWF: power failure when power < 3500W or > 9000W
    - HDF: heat dissipation failure when temp_diff < 8.6K and speed < 1380 rpm
    - OSF: overstrain failure when wear × torque exceeds variant-specific thresholds

    Args:
        df: DataFrame with original sensor columns.

    Returns:
        DataFrame with additional physics-based features.
    """
    df = df.copy()

    # Power (W) = Torque (Nm) × Angular velocity (rad/s)
    df["power_w"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)

    # Temperature difference
    df["temp_diff_k"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Wear-torque interaction
    df["wear_torque_product"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    # Torque-speed ratio (mechanical stress proxy)
    df["torque_speed_ratio"] = df["Torque [Nm]"] / (
        df["Rotational speed [rpm]"] + 1e-8
    )

    # Distance from PWF thresholds (normalized)
    df["power_low_margin"] = df["power_w"] - 3500
    df["power_high_margin"] = 9000 - df["power_w"]

    # Distance from HDF threshold
    df["temp_diff_margin"] = df["temp_diff_k"] - 8.6

    # Normalized wear (tool wear relative to typical failure range 200-240 min)
    df["wear_fraction"] = df["Tool wear [min]"] / 240.0

    return df


def get_feature_columns(include_physics: bool = True) -> list[str]:
    """Get the full list of feature column names after engineering.

    Args:
        include_physics: Whether to include physics-based derived features.

    Returns:
        List of feature column names.
    """
    base_cols = list(SENSOR_COLUMNS)
    type_cols = ["type_H", "type_L", "type_M"]

    if not include_physics:
        return base_cols + type_cols

    physics_cols = [
        "power_w", "temp_diff_k", "wear_torque_product", "torque_speed_ratio",
        "power_low_margin", "power_high_margin", "temp_diff_margin", "wear_fraction",
    ]

    return base_cols + type_cols + physics_cols


def prepare_for_training(
    df: pd.DataFrame,
    include_physics: bool = True,
) -> pd.DataFrame:
    """Apply all feature engineering and return a clean feature DataFrame.

    Args:
        df: Raw AI4I DataFrame.
        include_physics: Whether to add physics features.

    Returns:
        Feature-engineered DataFrame ready for model training.
    """
    df = df.copy()

    # One-hot encode Type
    type_dummies = pd.get_dummies(df["Type"], prefix="type")
    df = pd.concat([df, type_dummies], axis=1)

    # Add physics features
    if include_physics:
        df = add_physics_features(df)

    return df
