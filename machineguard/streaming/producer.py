"""Replay-based stream producer for MachineGuard.

Reads historical sensor data from C-MAPSS and AI4I datasets and publishes
each reading to Redis Streams at a configurable rate, simulating real-time
sensor telemetry from a fleet of machines.

Design: This is the production-grade replay simulator pattern — the same
one used by Uber, Netflix, and Google for backtesting ML pipelines. Real
sensor feeds would replace this module with a Kafka/Redpanda producer.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator

import pandas as pd
import redis

from machineguard.config import settings

logger = logging.getLogger(__name__)


class StreamProducer:
    """Publishes sensor readings to Redis Streams.

    Reads historical data and replays it at configurable speed.
    Each message is a flat dict of sensor values + metadata.
    """

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        replay_speed_ms: int | None = None,
    ):
        self.redis = redis_client or redis.Redis(
            host=settings.redis.redis_host,
            port=settings.redis.redis_port,
            db=settings.redis.redis_db,
            decode_responses=True,
        )
        self.speed_ms = replay_speed_ms or settings.streaming.replay_speed_ms
        self._running = True

    def stop(self) -> None:
        """Signal the producer to stop."""
        self._running = False

    # ---- C-MAPSS Replay ----

    def _load_cmapss_readings(
        self,
        subset: str = "FD001",
        max_engines: int | None = None,
    ) -> Iterator[dict]:
        """Yield C-MAPSS sensor readings as flat dicts."""
        from machineguard.data.cmapss_loader import (
            load_cmapss_raw,
            get_sensor_columns,
        )

        df = load_cmapss_raw(subset=subset)

        sensor_cols = get_sensor_columns(drop_constant=True)
        engines = df["unit_id"].unique()

        if max_engines:
            engines = engines[:max_engines]

        for unit_id in engines:
            unit_data = df[df["unit_id"] == unit_id].sort_values("cycle")

            for _, row in unit_data.iterrows():
                reading = {
                    "machine_id": f"cmapss_{subset}_{int(row['unit_id'])}",
                    "fleet_id": "cmapss",
                    "timestamp": int(row["cycle"]),
                    "cycle": int(row["cycle"]),
                }
                # Pack sensor values
                for col in sensor_cols:
                    reading[col] = float(row[col])

                yield reading

    def replay_cmapss(
        self,
        subset: str = "FD001",
        max_engines: int = 5,
        stream_name: str | None = None,
    ) -> int:
        """Replay C-MAPSS engine data to Redis Stream.

        Args:
            subset: C-MAPSS subset (FD001–FD004).
            max_engines: Number of engines to replay.
            stream_name: Override stream name.

        Returns:
            Total messages published.
        """
        stream = stream_name or settings.streaming.stream_cmapss
        count = 0

        logger.info(
            f"Starting C-MAPSS replay: {subset}, {max_engines} engines, "
            f"{self.speed_ms}ms/reading → stream '{stream}'"
        )

        for reading in self._load_cmapss_readings(subset, max_engines):
            if not self._running:
                logger.info("Producer stopped.")
                break

            self.redis.xadd(stream, reading, maxlen=50000)
            count += 1

            if count % 100 == 0:
                logger.debug(
                    f"Published {count} readings "
                    f"(machine={reading['machine_id']}, cycle={reading['cycle']})"
                )

            time.sleep(self.speed_ms / 1000.0)

        logger.info(f"C-MAPSS replay complete: {count} messages published.")
        return count

    # ---- AI4I Replay ----

    def _load_ai4i_readings(
        self,
        max_rows: int | None = None,
    ) -> Iterator[dict]:
        """Yield AI4I sensor readings as flat dicts."""
        from machineguard.data.ai4i_loader import load_ai4i_raw

        df = load_ai4i_raw()

        if max_rows:
            df = df.head(max_rows)

        sensor_cols = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

        for idx, row in df.iterrows():
            reading = {
                "machine_id": f"ai4i_{row.get('Product ID', idx)}",
                "fleet_id": "ai4i",
                "timestamp": int(idx),
            }
            for col in sensor_cols:
                reading[col] = float(row[col])

            # Include product type for operating conditions
            if "Type" in row:
                reading["product_type"] = str(row["Type"])

            yield reading

    def replay_ai4i(
        self,
        max_rows: int = 500,
        stream_name: str | None = None,
    ) -> int:
        """Replay AI4I manufacturing data to Redis Stream.

        Args:
            max_rows: Max number of rows to replay.
            stream_name: Override stream name.

        Returns:
            Total messages published.
        """
        stream = stream_name or settings.streaming.stream_ai4i
        count = 0

        logger.info(
            f"Starting AI4I replay: {max_rows} rows, "
            f"{self.speed_ms}ms/reading → stream '{stream}'"
        )

        for reading in self._load_ai4i_readings(max_rows):
            if not self._running:
                logger.info("Producer stopped.")
                break

            self.redis.xadd(stream, reading, maxlen=50000)
            count += 1

            if count % 100 == 0:
                logger.debug(f"Published {count} AI4I readings")

            time.sleep(self.speed_ms / 1000.0)

        logger.info(f"AI4I replay complete: {count} messages published.")
        return count

    def replay_both(
        self,
        cmapss_engines: int = 5,
        ai4i_rows: int = 500,
    ) -> dict[str, int]:
        """Replay both fleets sequentially.

        Returns:
            Dict with counts per fleet.
        """
        results = {}
        results["cmapss"] = self.replay_cmapss(max_engines=cmapss_engines)
        results["ai4i"] = self.replay_ai4i(max_rows=ai4i_rows)
        return results
