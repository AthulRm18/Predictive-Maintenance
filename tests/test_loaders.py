"""Tests for data loaders — verify schema compliance and edge cases."""

import pytest
import numpy as np
import pandas as pd

from machineguard.data.schema import MachineReading


class TestMachineReading:
    """Test the common MachineReading schema."""

    def test_create_reading(self):
        reading = MachineReading(
            machine_id="test_unit_1",
            fleet_id="test_fleet",
            timestamp=42,
            sensor_readings={"sensor_1": 100.0, "sensor_2": 200.0},
            operating_conditions={"mode": "normal"},
            label={"rul": 50},
        )
        assert reading.machine_id == "test_unit_1"
        assert reading.fleet_id == "test_fleet"
        assert reading.timestamp == 42
        assert reading.sensor_readings["sensor_1"] == 100.0
        assert reading.label["rul"] == 50

    def test_to_dict(self):
        reading = MachineReading(
            machine_id="test_1",
            fleet_id="fleet_a",
            timestamp=1,
            sensor_readings={"temp": 300.0, "pressure": 1.0},
            label={"failure": 0},
        )
        d = reading.to_dict()
        assert d["machine_id"] == "test_1"
        assert d["sensor_temp"] == 300.0
        assert d["sensor_pressure"] == 1.0
        assert d["label_failure"] == 0

    def test_to_dict_no_label(self):
        reading = MachineReading(
            machine_id="test_2",
            fleet_id="fleet_b",
            timestamp=2,
            sensor_readings={"s1": 1.0},
        )
        d = reading.to_dict()
        assert "label_rul" not in d

    def test_to_dataframe(self):
        readings = [
            MachineReading(
                machine_id=f"unit_{i}",
                fleet_id="test",
                timestamp=i,
                sensor_readings={"s1": float(i), "s2": float(i * 2)},
                label={"rul": 100 - i},
            )
            for i in range(5)
        ]
        df = MachineReading.to_dataframe(readings)
        assert len(df) == 5
        assert "sensor_s1" in df.columns
        assert "sensor_s2" in df.columns
        assert "label_rul" in df.columns
        assert df["sensor_s1"].iloc[3] == 3.0


class TestCMAPSSLoader:
    """Test C-MAPSS loader functions (unit tests that don't require data files)."""

    def test_get_sensor_columns_with_constant_drop(self):
        from machineguard.data.cmapss_loader import get_sensor_columns, CONSTANT_SENSORS_FD001

        cols = get_sensor_columns(drop_constant=True)
        for const_sensor in CONSTANT_SENSORS_FD001:
            assert const_sensor not in cols
        assert len(cols) == 21 - len(CONSTANT_SENSORS_FD001)

    def test_get_sensor_columns_all(self):
        from machineguard.data.cmapss_loader import get_sensor_columns

        cols = get_sensor_columns(drop_constant=False)
        assert len(cols) == 21

    def test_compute_rul(self):
        from machineguard.data.cmapss_loader import compute_rul

        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
        })
        result = compute_rul(df, rul_cap=None)
        assert result["rul"].tolist() == [2, 1, 0, 1, 0]

    def test_compute_rul_with_cap(self):
        from machineguard.data.cmapss_loader import compute_rul

        df = pd.DataFrame({
            "unit_id": [1] * 200,
            "cycle": list(range(1, 201)),
        })
        result = compute_rul(df, rul_cap=125)
        assert result["rul"].max() == 125
        assert result["rul"].min() == 0


class TestAI4ILoader:
    """Test AI4I loader functions."""

    def test_resolve_failure_mode_no_failure(self):
        from machineguard.data.ai4i_loader import resolve_failure_mode

        row = pd.Series({
            "Machine failure": 0,
            "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
        })
        assert resolve_failure_mode(row) == "no_failure"

    def test_resolve_failure_mode_single(self):
        from machineguard.data.ai4i_loader import resolve_failure_mode

        row = pd.Series({
            "Machine failure": 1,
            "TWF": 0, "HDF": 1, "PWF": 0, "OSF": 0, "RNF": 0,
        })
        assert resolve_failure_mode(row) == "HDF"

    def test_resolve_failure_mode_multi_picks_rarest(self):
        from machineguard.data.ai4i_loader import resolve_failure_mode

        row = pd.Series({
            "Machine failure": 1,
            "TWF": 1, "HDF": 1, "PWF": 0, "OSF": 0, "RNF": 1,
        })
        # RNF is rarest (rank 0), should be selected
        assert resolve_failure_mode(row) == "RNF"

    def test_add_derived_features(self):
        from machineguard.data.ai4i_loader import add_derived_features

        df = pd.DataFrame({
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [100.0],
        })
        result = add_derived_features(df)
        assert "power_w" in result.columns
        assert "temp_diff_k" in result.columns
        assert "wear_torque_product" in result.columns
        assert result["temp_diff_k"].iloc[0] == 10.0
        assert result["wear_torque_product"].iloc[0] == 4000.0
