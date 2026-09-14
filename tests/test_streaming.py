"""Tests for streaming components.

Tests the RollingBuffer, StreamProducer (data loading), and StreamConsumer
logic WITHOUT requiring a running Redis instance.
"""

import pytest
import pandas as pd
import numpy as np

from machineguard.streaming.consumer import RollingBuffer


class TestRollingBuffer:
    """Test the per-machine rolling buffer."""

    def test_add_and_size(self):
        buf = RollingBuffer(max_size=5)
        assert buf.size == 0
        assert not buf.is_ready

        for i in range(3):
            buf.add({"sensor_1": float(i), "cycle": i})

        assert buf.size == 3
        assert not buf.is_ready  # Need 5 for is_ready

    def test_is_ready(self):
        buf = RollingBuffer(max_size=10)
        for i in range(5):
            buf.add({"sensor_1": float(i)})

        assert buf.is_ready  # Minimum 5 readings

    def test_max_size_eviction(self):
        buf = RollingBuffer(max_size=3)
        for i in range(10):
            buf.add({"val": float(i)})

        assert buf.size == 3
        df = buf.to_dataframe()
        # Should contain the last 3 readings: 7, 8, 9
        assert df["val"].tolist() == [7.0, 8.0, 9.0]

    def test_to_dataframe(self):
        buf = RollingBuffer(max_size=5)
        for i in range(3):
            buf.add({"sensor_1": float(i * 10), "sensor_2": float(i * 20)})

        df = buf.to_dataframe()
        assert len(df) == 3
        assert "sensor_1" in df.columns
        assert "sensor_2" in df.columns
        assert df["sensor_1"].iloc[2] == 20.0


class TestProducerDataLoading:
    """Test producer's data loading logic (no Redis needed)."""

    def test_cmapss_readings_iterator(self):
        """Test that C-MAPSS readings are yielded correctly."""
        from machineguard.streaming.producer import StreamProducer
        from machineguard.config import DATA_DIR

        cmapss_path = DATA_DIR / "cmapss" / "train_FD001.txt"
        if not cmapss_path.exists():
            pytest.skip("C-MAPSS data not downloaded")

        try:
            producer = StreamProducer.__new__(StreamProducer)
            readings = list(producer._load_cmapss_readings(max_engines=1))
        except FileNotFoundError:
            pytest.skip("C-MAPSS data not available")

        assert len(readings) > 0
        first = readings[0]
        assert "machine_id" in first
        assert "fleet_id" in first
        assert first["fleet_id"] == "cmapss"
        assert "cycle" in first
        sensor_keys = [k for k in first if k.startswith("sensor_")]
        assert len(sensor_keys) > 5

    def test_ai4i_readings_iterator(self):
        """Test that AI4I readings are yielded correctly."""
        from machineguard.streaming.producer import StreamProducer
        from machineguard.config import DATA_DIR

        ai4i_path = DATA_DIR / "ai4i" / "ai4i2020.csv"
        if not ai4i_path.exists():
            pytest.skip("AI4I data not downloaded")

        try:
            producer = StreamProducer.__new__(StreamProducer)
            readings = list(producer._load_ai4i_readings(max_rows=10))
        except FileNotFoundError:
            pytest.skip("AI4I data not available")

        assert len(readings) == 10
        first = readings[0]
        assert "machine_id" in first
        assert first["fleet_id"] == "ai4i"
        assert "Air temperature [K]" in first
        assert "Torque [Nm]" in first


class TestWebSocketBroadcaster:
    """Test the PredictionBroadcaster logic."""

    def test_connection_count(self):
        from machineguard.streaming.websocket_handler import PredictionBroadcaster

        b = PredictionBroadcaster()
        assert b.connection_count == 0

    def test_push_prediction(self):
        """Test that push_prediction doesn't error without queue."""
        from machineguard.streaming.websocket_handler import PredictionBroadcaster

        b = PredictionBroadcaster()
        # Should not raise even without queue
        b.push_prediction({"type": "rul", "machine_id": "test"})
