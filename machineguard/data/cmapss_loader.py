"""C-MAPSS Turbofan Engine Degradation dataset loader.

Loads NASA C-MAPSS FD001 (default) run-to-failure data and normalizes it
into the common MachineReading schema.

Dataset format (space-delimited, no headers):
    Columns: unit_id, cycle, op_setting_1, op_setting_2, op_setting_3,
             sensor_1, sensor_2, ..., sensor_21

Training set contains full run-to-failure trajectories.
Test set contains truncated trajectories with true RUL in a separate file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from machineguard.config import DATA_DIR, settings
from machineguard.data.schema import MachineReading

# Column names for C-MAPSS (26 columns total)
CMAPSS_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors that are essentially constant in FD001 (near-zero variance)
# These carry no degradation information and should be dropped for modeling
CONSTANT_SENSORS_FD001 = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                          "sensor_16", "sensor_18", "sensor_19"]

# Sensor names for reference
SENSOR_NAMES = {
    "sensor_2": "LPC outlet temperature",
    "sensor_3": "HPC outlet temperature",
    "sensor_4": "LPT outlet temperature",
    "sensor_7": "Total temperature at HPC outlet",
    "sensor_8": "Physical fan speed",
    "sensor_9": "Physical core speed",
    "sensor_11": "Static pressure at HPC outlet",
    "sensor_12": "Ratio of fuel flow to Ps30",
    "sensor_13": "Corrected fan speed",
    "sensor_14": "Corrected core speed",
    "sensor_15": "Bypass ratio",
    "sensor_17": "Bleed enthalpy",
    "sensor_20": "HPT coolant bleed",
    "sensor_21": "LPT coolant bleed",
}


def load_cmapss_raw(
    subset: str = "FD001",
    split: Literal["train", "test"] = "train",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load raw C-MAPSS data into a DataFrame with named columns.

    Args:
        subset: Which subset to load (FD001, FD002, FD003, FD004).
        split: "train" for run-to-failure, "test" for truncated trajectories.
        data_dir: Override data directory path.

    Returns:
        DataFrame with columns: unit_id, cycle, op_setting_1-3, sensor_1-21
    """
    base = data_dir or (DATA_DIR / "cmapss")
    filepath = base / f"{split}_{subset}.txt"

    if not filepath.exists():
        raise FileNotFoundError(
            f"C-MAPSS file not found: {filepath}\n"
            f"Run 'python scripts/download_datasets.py' first."
        )

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
        engine="python",
    )

    return df


def compute_rul(df: pd.DataFrame, rul_cap: int | None = None) -> pd.DataFrame:
    """Compute Remaining Useful Life (RUL) for training data.

    For each engine unit, RUL = max_cycle - current_cycle.
    Optionally caps RUL at a maximum value (piece-wise linear degradation model).

    The cap (default 125) avoids modeling the flat early-life region where
    the engine is healthy and RUL differences are not meaningful. This is
    standard practice in the C-MAPSS literature.

    Args:
        df: Raw C-MAPSS training DataFrame.
        rul_cap: Maximum RUL value. None = no cap. Default uses settings.

    Returns:
        DataFrame with added 'rul' column.
    """
    if rul_cap is None:
        rul_cap = settings.model.rul_cap

    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].max()
    df["rul"] = df.apply(
        lambda row: max_cycles[row["unit_id"]] - row["cycle"], axis=1
    )

    if rul_cap is not None:
        df["rul"] = df["rul"].clip(upper=rul_cap)

    return df


def load_rul_targets(
    subset: str = "FD001",
    data_dir: Path | None = None,
) -> pd.Series:
    """Load true RUL values for test data.

    Args:
        subset: Which subset (FD001-FD004).
        data_dir: Override data directory.

    Returns:
        Series indexed by unit_id (1-based) with true RUL values.
    """
    base = data_dir or (DATA_DIR / "cmapss")
    filepath = base / f"RUL_{subset}.txt"

    if not filepath.exists():
        raise FileNotFoundError(f"RUL file not found: {filepath}")

    rul = pd.read_csv(filepath, header=None, names=["rul"])
    rul.index = rul.index + 1  # 1-based unit_id
    rul.index.name = "unit_id"
    return rul["rul"]


def get_sensor_columns(drop_constant: bool = True, subset: str = "FD001") -> list[str]:
    """Get the list of informative sensor column names.

    Args:
        drop_constant: If True, removes sensors with near-zero variance in FD001.
        subset: Which subset (only affects constant sensor filtering).

    Returns:
        List of sensor column names.
    """
    all_sensors = [f"sensor_{i}" for i in range(1, 22)]

    if drop_constant and subset == "FD001":
        return [s for s in all_sensors if s not in CONSTANT_SENSORS_FD001]

    return all_sensors


def get_op_columns() -> list[str]:
    """Get operating condition column names."""
    return [f"op_setting_{i}" for i in range(1, 4)]


def normalize_sensors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    sensor_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Normalize sensor readings using MinMax scaling.

    Fits on training data only to prevent data leakage.

    Args:
        train_df: Training DataFrame.
        test_df: Optional test DataFrame to transform.
        sensor_cols: Columns to normalize. Defaults to informative sensors.

    Returns:
        Tuple of (normalized_train, normalized_test_or_None, scaler_params).
    """
    if sensor_cols is None:
        sensor_cols = get_sensor_columns()

    train_df = train_df.copy()
    scaler_params = {}

    for col in sensor_cols:
        col_min = train_df[col].min()
        col_max = train_df[col].max()
        col_range = col_max - col_min

        if col_range == 0:
            # Constant column — set to 0
            train_df[col] = 0.0
            scaler_params[col] = {"min": col_min, "max": col_max, "range": 0}
        else:
            train_df[col] = (train_df[col] - col_min) / col_range
            scaler_params[col] = {"min": col_min, "max": col_max, "range": col_range}

    normalized_test = None
    if test_df is not None:
        normalized_test = test_df.copy()
        for col in sensor_cols:
            params = scaler_params[col]
            if params["range"] == 0:
                normalized_test[col] = 0.0
            else:
                normalized_test[col] = (
                    (normalized_test[col] - params["min"]) / params["range"]
                )

    return train_df, normalized_test, scaler_params


def to_machine_readings(
    df: pd.DataFrame,
    subset: str = "FD001",
    has_rul: bool = True,
) -> list[MachineReading]:
    """Convert a C-MAPSS DataFrame to a list of MachineReading objects.

    Args:
        df: C-MAPSS DataFrame with named columns (and optionally 'rul').
        subset: Subset identifier for the machine_id prefix.
        has_rul: Whether the DataFrame has a 'rul' column.

    Returns:
        List of MachineReading objects in common schema.
    """
    readings = []
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]

    for _, row in df.iterrows():
        unit_id = int(row["unit_id"])

        reading = MachineReading(
            machine_id=f"cmapss_{subset.lower()}_unit_{unit_id}",
            fleet_id="cmapss",
            timestamp=int(row["cycle"]),
            sensor_readings={col: float(row[col]) for col in sensor_cols},
            operating_conditions={col: float(row[col]) for col in op_cols},
            label={"rul": int(row["rul"])} if has_rul and "rul" in row.index else None,
        )
        readings.append(reading)

    return readings


def load_train_test(
    subset: str = "FD001",
    normalize: bool = True,
    drop_constant_sensors: bool = True,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load, process, and return train/test data for modeling.

    This is the main entry point for the C-MAPSS data pipeline.

    Args:
        subset: Which C-MAPSS subset (FD001-FD004).
        normalize: Whether to MinMax-normalize sensor readings.
        drop_constant_sensors: Whether to drop constant sensors.
        data_dir: Override data directory.

    Returns:
        Tuple of (train_df_with_rul, test_df, true_rul_series).
        Train DataFrame has an added 'rul' column.
    """
    train_df = load_cmapss_raw(subset, "train", data_dir)
    test_df = load_cmapss_raw(subset, "test", data_dir)
    true_rul = load_rul_targets(subset, data_dir)

    # Add RUL labels to training data
    train_df = compute_rul(train_df)

    # Normalize if requested
    if normalize:
        sensor_cols = get_sensor_columns(drop_constant_sensors, subset)
        train_df, test_df, _ = normalize_sensors(train_df, test_df, sensor_cols)

    return train_df, test_df, true_rul
