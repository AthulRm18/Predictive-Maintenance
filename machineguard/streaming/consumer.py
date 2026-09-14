"""Stream consumer with rolling feature computation and real-time prediction.

Consumes sensor readings from Redis Streams, maintains a per-machine rolling
window buffer, computes features on-the-fly, runs predictions against loaded
models, and publishes results via WebSocket-compatible queue.

This is the core "live inference" loop of the system.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from typing import Any, Callable

import numpy as np
import pandas as pd
import redis

from machineguard.config import settings, MODELS_DIR

logger = logging.getLogger(__name__)


class RollingBuffer:
    """Per-machine rolling window of sensor readings.

    Maintains the last N readings for a machine and can produce a
    DataFrame suitable for feature computation on demand.
    """

    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.readings: deque[dict] = deque(maxlen=max_size)

    def add(self, reading: dict) -> None:
        self.readings.append(reading)

    @property
    def size(self) -> int:
        return len(self.readings)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert buffer to DataFrame for feature computation."""
        return pd.DataFrame(list(self.readings))

    @property
    def is_ready(self) -> bool:
        """Whether we have enough readings for reliable features."""
        return self.size >= 5  # Minimum for rolling window


class StreamConsumer:
    """Consumes Redis Streams, computes features, runs predictions.

    Maintains a rolling buffer per machine. When enough readings
    accumulate, computes features and runs inference.
    """

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        on_prediction: Callable[[dict], None] | None = None,
    ):
        self.redis = redis_client or redis.Redis(
            host=settings.redis.redis_host,
            port=settings.redis.redis_port,
            db=settings.redis.redis_db,
            decode_responses=True,
        )
        self.buffers: dict[str, RollingBuffer] = defaultdict(
            lambda: RollingBuffer(settings.streaming.rolling_buffer_size)
        )
        self.on_prediction = on_prediction
        self._running = True
        self._rul_model = None
        self._fault_model = None
        self._prediction_count = 0

    def stop(self) -> None:
        """Signal the consumer to stop."""
        self._running = False

    def _load_models(self) -> None:
        """Load models for inference."""
        from machineguard.models.rul_model import RULModel
        from machineguard.models.fault_classifier import FaultClassifier

        rul_path = MODELS_DIR / "rul_model"
        if (rul_path / "model.joblib").exists():
            self._rul_model = RULModel()
            self._rul_model.load(rul_path)
            logger.info("Stream consumer: RUL model loaded.")

        fault_path = MODELS_DIR / "fault_classifier"
        if (fault_path / "model.joblib").exists():
            self._fault_model = FaultClassifier()
            self._fault_model.load(fault_path)
            logger.info("Stream consumer: Fault model loaded.")

    def _predict_cmapss(self, machine_id: str, buffer: RollingBuffer) -> dict | None:
        """Run RUL prediction on C-MAPSS readings."""
        if self._rul_model is None:
            return None

        try:
            df = buffer.to_dataframe()
            # Use the latest reading's sensor values
            latest = df.iloc[-1]
            sensor_readings = {
                col: float(latest[col])
                for col in latest.index
                if col.startswith("sensor_")
            }

            result = self._rul_model.predict(sensor_readings)
            return {
                "type": "rul",
                "machine_id": machine_id,
                "fleet_id": "cmapss",
                "predicted_rul": result["predicted_rul"],
                "rul_cap": result["rul_cap"],
                "cycle": int(latest.get("cycle", 0)),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.warning(f"RUL prediction failed for {machine_id}: {e}")
            return None

    def _predict_ai4i(self, machine_id: str, buffer: RollingBuffer) -> dict | None:
        """Run fault classification on AI4I readings."""
        if self._fault_model is None:
            return None

        try:
            df = buffer.to_dataframe()
            latest = df.iloc[-1]

            # Build sensor readings dict
            sensor_cols = [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
            ]
            sensor_readings = {
                col: float(latest[col])
                for col in sensor_cols
                if col in latest.index
            }

            result = self._fault_model.predict(sensor_readings)
            return {
                "type": "fault",
                "machine_id": machine_id,
                "fleet_id": "ai4i",
                "predicted_class": result["predicted_class"],
                "probabilities": result["probabilities"],
                "anomaly_score": result["anomaly_score"],
                "confidence": result["confidence"],
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.warning(f"Fault prediction failed for {machine_id}: {e}")
            return None

    def _process_reading(self, stream_id: str, data: dict) -> dict | None:
        """Process a single sensor reading and optionally produce a prediction."""
        machine_id = data.get("machine_id", "unknown")
        fleet_id = data.get("fleet_id", "unknown")

        # Add to rolling buffer
        self.buffers[machine_id].add(data)
        buffer = self.buffers[machine_id]

        if not buffer.is_ready:
            return None

        # Route to appropriate model based on fleet
        prediction = None
        if fleet_id == "cmapss":
            prediction = self._predict_cmapss(machine_id, buffer)
        elif fleet_id == "ai4i":
            prediction = self._predict_ai4i(machine_id, buffer)

        if prediction:
            self._prediction_count += 1

            # Publish prediction to Redis stream for WebSocket consumers
            self.redis.xadd(
                settings.streaming.stream_predictions,
                {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in prediction.items()},
                maxlen=10000,
            )

            # Callback for WebSocket push
            if self.on_prediction:
                self.on_prediction(prediction)

            if self._prediction_count % 50 == 0:
                logger.info(
                    f"Predictions made: {self._prediction_count} "
                    f"(latest: {machine_id} → {prediction.get('type')})"
                )

        return prediction

    def consume(
        self,
        streams: list[str] | None = None,
        timeout_ms: int = 1000,
        max_iterations: int | None = None,
    ) -> int:
        """Main consumption loop.

        Reads from Redis Streams and processes each reading.

        Args:
            streams: List of stream names to consume from.
            timeout_ms: Block timeout for XREAD.
            max_iterations: Max number of read cycles (None = infinite).

        Returns:
            Total predictions made.
        """
        if streams is None:
            streams = [
                settings.streaming.stream_cmapss,
                settings.streaming.stream_ai4i,
            ]

        self._load_models()

        # Track last seen ID per stream
        last_ids = {s: "0-0" for s in streams}
        iterations = 0

        logger.info(f"Stream consumer started. Listening on: {streams}")

        while self._running:
            if max_iterations and iterations >= max_iterations:
                break

            try:
                # XREAD: read new messages from all streams
                stream_entries = {s: last_ids[s] for s in streams}
                results = self.redis.xread(
                    stream_entries,
                    count=settings.streaming.batch_size,
                    block=timeout_ms,
                )

                if results:
                    for stream_name, messages in results:
                        # Handle bytes vs str from redis
                        if isinstance(stream_name, bytes):
                            stream_name = stream_name.decode()

                        for msg_id, data in messages:
                            if isinstance(msg_id, bytes):
                                msg_id = msg_id.decode()
                            # Decode bytes in data if needed
                            decoded = {}
                            for k, v in data.items():
                                key = k.decode() if isinstance(k, bytes) else k
                                val = v.decode() if isinstance(v, bytes) else v
                                # Try to convert numeric strings
                                try:
                                    decoded[key] = float(val)
                                except (ValueError, TypeError):
                                    decoded[key] = val

                            self._process_reading(msg_id, decoded)
                            last_ids[stream_name] = msg_id

                iterations += 1

            except redis.ConnectionError as e:
                logger.error(f"Redis connection error: {e}. Retrying in 2s...")
                time.sleep(2)
            except Exception as e:
                logger.error(f"Consumer error: {e}", exc_info=True)
                time.sleep(1)

        logger.info(
            f"Consumer stopped. Total predictions: {self._prediction_count}"
        )
        return self._prediction_count

    @property
    def stats(self) -> dict:
        """Return consumer statistics."""
        return {
            "total_predictions": self._prediction_count,
            "active_machines": len(self.buffers),
            "buffer_sizes": {
                mid: buf.size for mid, buf in self.buffers.items()
            },
        }
