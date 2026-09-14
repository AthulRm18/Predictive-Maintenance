"""AI4I 2020 Predictive Maintenance dataset loader.

Loads the UCI AI4I 2020 synthetic predictive maintenance dataset and
normalizes it into the common MachineReading schema.

Dataset: 10,000 data points with 6 features and 5 failure modes.
Features: Type (L/M/H), Air temp [K], Process temp [K], Rotational speed [rpm],
          Torque [Nm], Tool wear [min]
Failure modes: TWF, HDF, PWF, OSF, RNF
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from machineguard.config import DATA_DIR
from machineguard.data.schema import MachineReading

# Column mappings
SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Standardized sensor names for common schema
SENSOR_NAME_MAP = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Rarity ranking (rarest first) — used for multi-label → multi-class resolution
# Based on dataset documentation: RNF(5) < TWF(51) < PWF(95) < OSF(98) < HDF(115)
FAILURE_RARITY = {"RNF": 0, "TWF": 1, "PWF": 2, "OSF": 3, "HDF": 4}


def load_ai4i_raw(data_dir: Path | None = None) -> pd.DataFrame:
    """Load raw AI4I 2020 CSV data.

    Args:
        data_dir: Override data directory path.

    Returns:
        Raw DataFrame as loaded from CSV.
    """
    base = data_dir or (DATA_DIR / "ai4i")

    # Try multiple possible filenames
    possible_names = ["ai4i2020.csv", "ai4i_2020.csv", "predictive_maintenance.csv"]
    filepath = None
    for name in possible_names:
        candidate = base / name
        if candidate.exists():
            filepath = candidate
            break

    if filepath is None:
        raise FileNotFoundError(
            f"AI4I dataset not found in {base}\n"
            f"Expected one of: {possible_names}\n"
            f"Run 'python scripts/download_datasets.py' first."
        )

    df = pd.read_csv(filepath)
    return df


def resolve_failure_mode(row: pd.Series) -> str:
    """Resolve multi-label failure modes to a single class.

    When multiple failure modes are active simultaneously, assigns the
    rarest mode. This gives better signal for imbalanced learning since
    rare classes get priority.

    Args:
        row: DataFrame row with failure mode columns.

    Returns:
        Failure mode string or "no_failure".
    """
    if row["Machine failure"] == 0:
        return "no_failure"

    active_modes = [mode for mode in FAILURE_MODES if row.get(mode, 0) == 1]

    if not active_modes:
        # Machine failure = 1 but no specific mode flagged (shouldn't happen)
        return "unknown_failure"

    # Return the rarest active mode
    return min(active_modes, key=lambda m: FAILURE_RARITY.get(m, 99))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-based derived features.

    These features mirror the actual failure mode physics:
    - Power: torque × angular_velocity — PWF triggers when < 3500W or > 9000W
    - Temperature difference: process - air temp — HDF triggers when < 8.6K
    - Wear-torque product: tool_wear × torque — OSF triggers at thresholds

    Args:
        df: DataFrame with sensor columns.

    Returns:
        DataFrame with added derived feature columns.
    """
    df = df.copy()

    # Power (W) = Torque (Nm) × Angular velocity (rad/s)
    # Angular velocity = RPM × 2π/60
    df["power_w"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)

    # Temperature difference (K)
    df["temp_diff_k"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Wear-torque interaction
    df["wear_torque_product"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    # Torque-speed ratio (proxy for mechanical stress)
    df["torque_speed_ratio"] = df["Torque [Nm]"] / (
        df["Rotational speed [rpm]"] + 1e-8  # avoid div by zero
    )

    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    include_derived: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare feature matrix and target vectors for modeling.

    Args:
        df: Raw AI4I DataFrame.
        include_derived: Whether to add physics-based derived features.

    Returns:
        Tuple of (X_features, y_binary, y_multiclass).
        y_binary: 0/1 machine failure.
        y_multiclass: string failure mode class.
    """
    df = df.copy()

    # Add derived features
    if include_derived:
        df = add_derived_features(df)

    # Resolve multi-class target
    df["failure_mode"] = df.apply(resolve_failure_mode, axis=1)

    # Build feature matrix
    feature_cols = list(SENSOR_COLUMNS)

    # One-hot encode Type
    type_dummies = pd.get_dummies(df["Type"], prefix="type")
    df = pd.concat([df, type_dummies], axis=1)
    feature_cols.extend(type_dummies.columns.tolist())

    # Add derived features if present
    derived_cols = ["power_w", "temp_diff_k", "wear_torque_product", "torque_speed_ratio"]
    if include_derived:
        feature_cols.extend(derived_cols)

    X = df[feature_cols].copy()
    y_binary = df["Machine failure"].copy()
    y_multiclass = df["failure_mode"].copy()

    return X, y_binary, y_multiclass


def to_machine_readings(df: pd.DataFrame) -> list[MachineReading]:
    """Convert AI4I DataFrame to list of MachineReading objects.

    Args:
        df: Raw AI4I DataFrame (before feature engineering).

    Returns:
        List of MachineReading objects in common schema.
    """
    readings = []

    for idx, row in df.iterrows():
        # Build sensor readings dict with standardized names
        sensor_dict = {}
        for orig_name, std_name in SENSOR_NAME_MAP.items():
            if orig_name in row.index:
                sensor_dict[std_name] = float(row[orig_name])

        # Resolve failure mode
        failure_mode = resolve_failure_mode(row)

        # Build label
        label = {
            "failure": int(row.get("Machine failure", 0)),
            "failure_mode": failure_mode,
        }
        # Include individual mode flags
        for mode in FAILURE_MODES:
            if mode in row.index:
                label[mode.lower()] = int(row[mode])

        reading = MachineReading(
            machine_id=f"ai4i_{row.get('Product ID', f'unknown_{idx}')}",
            fleet_id="ai4i",
            timestamp=int(idx),
            sensor_readings=sensor_dict,
            operating_conditions={"type": str(row.get("Type", "unknown"))},
            label=label,
        )
        readings.append(reading)

    return readings


def load_prepared(
    data_dir: Path | None = None,
    include_derived: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Load, process, and split AI4I data for modeling.

    This is the main entry point for the AI4I data pipeline.

    Args:
        data_dir: Override data directory.
        include_derived: Whether to add physics-based features.
        test_size: Fraction of data for test set.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with keys: X_train, X_test, y_train_binary, y_test_binary,
        y_train_multi, y_test_multi, feature_names, class_names, raw_df.
    """
    from sklearn.model_selection import train_test_split

    raw_df = load_ai4i_raw(data_dir)
    X, y_binary, y_multiclass = prepare_features_and_target(raw_df, include_derived)

    # Stratified split on multiclass target
    # Some classes (RNF) may have < 2 samples, which breaks StratifiedShuffleSplit
    # in newer scikit-learn. Fall back to non-stratified split in that case.
    class_counts = y_multiclass.value_counts()
    can_stratify = (class_counts >= 2).all()

    X_train, X_test, y_train_b, y_test_b, y_train_m, y_test_m = train_test_split(
        X, y_binary, y_multiclass,
        test_size=test_size,
        random_state=random_state,
        stratify=y_multiclass if can_stratify else None,
    )

    # Class distribution
    class_names = sorted(y_multiclass.unique().tolist())

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_binary": y_train_b,
        "y_test_binary": y_test_b,
        "y_train_multi": y_train_m,
        "y_test_multi": y_test_m,
        "feature_names": X.columns.tolist(),
        "class_names": class_names,
        "raw_df": raw_df,
    }
