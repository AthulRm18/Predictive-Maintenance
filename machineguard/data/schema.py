"""Common data schema for MachineGuard.

All dataset loaders normalize their data into this schema, enabling
schema-agnostic downstream processing (feature engineering, inference,
fleet-level fingerprinting in Stage 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MachineReading:
    """A single sensor reading from any machine in any fleet.

    This is the schema-agnostic representation that both C-MAPSS and AI4I
    loaders produce. The sensor_readings dict can hold any number of sensors
    with any names — the downstream feature engineering layer handles the
    fleet-specific transformations.

    Attributes:
        machine_id: Unique identifier, e.g. "cmapss_fd001_unit_42" or "ai4i_L48732"
        fleet_id: Dataset family identifier — "cmapss" or "ai4i"
        timestamp: Cycle number (C-MAPSS) or sequential index (AI4I)
        sensor_readings: Dict of sensor name → value. Keys differ by fleet.
        operating_conditions: Dict of operating regime info.
            C-MAPSS: {"op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0}
            AI4I: {"type": "L"}
        label: Dict of target variables, or None if unlabeled.
            C-MAPSS: {"rul": 112}
            AI4I: {"failure": 1, "failure_mode": "HDF"}
    """

    machine_id: str
    fleet_id: str
    timestamp: int
    sensor_readings: dict[str, float]
    operating_conditions: dict[str, Any] = field(default_factory=dict)
    label: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dictionary for DataFrame construction."""
        result = {
            "machine_id": self.machine_id,
            "fleet_id": self.fleet_id,
            "timestamp": self.timestamp,
        }
        # Flatten sensor readings with prefix
        for key, value in self.sensor_readings.items():
            result[f"sensor_{key}"] = value
        # Flatten operating conditions
        for key, value in self.operating_conditions.items():
            result[f"op_{key}"] = value
        # Flatten labels
        if self.label:
            for key, value in self.label.items():
                result[f"label_{key}"] = value
        return result

    @staticmethod
    def to_dataframe(readings: list[MachineReading]) -> "pd.DataFrame":
        """Convert a list of MachineReading objects to a pandas DataFrame.

        This produces a wide-format DataFrame suitable for ML pipelines.
        Sensor columns are prefixed with 'sensor_', operating conditions
        with 'op_', and labels with 'label_'.
        """
        import pandas as pd
        records = [r.to_dict() for r in readings]
        return pd.DataFrame(records)
