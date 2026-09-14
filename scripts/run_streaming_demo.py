"""Streaming demo — run producer + consumer with live predictions.

One-command demo that replays sensor data through Redis Streams
and shows real-time predictions in the terminal.

Usage:
    python scripts/run_streaming_demo.py [--fleet cmapss|ai4i|both] [--speed-ms 50] [--engines 3]

Requirements:
    - Redis running on localhost:6379 (or docker-compose up -d redis)
    - Models trained (python scripts/train_models.py)
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from machineguard.streaming.producer import StreamProducer
from machineguard.streaming.consumer import StreamConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("streaming_demo")


def prediction_callback(prediction: dict) -> None:
    """Print each prediction to the terminal."""
    pred_type = prediction.get("type", "unknown")

    if pred_type == "rul":
        rul = prediction.get("predicted_rul", "?")
        machine = prediction.get("machine_id", "?")
        cycle = prediction.get("cycle", "?")
        print(
            f"  [RUL] {machine} | cycle={cycle} | "
            f"predicted_rul={rul:.1f} cycles remaining"
        )
    elif pred_type == "fault":
        cls = prediction.get("predicted_class", "?")
        machine = prediction.get("machine_id", "?")
        conf = prediction.get("confidence", 0)
        anomaly = prediction.get("anomaly_score", 0)
        print(
            f"  [FAULT] {machine} | "
            f"class={cls} | confidence={conf:.3f} | anomaly={anomaly:.3f}"
        )


def run_demo(
    fleet: str = "both",
    speed_ms: int = 50,
    max_engines: int = 3,
    max_ai4i_rows: int = 200,
) -> None:
    """Run the full streaming demo."""

    import redis as redis_lib

    # Check Redis connection
    try:
        r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        logger.info("Redis connection OK")
    except redis_lib.ConnectionError:
        logger.error(
            "Cannot connect to Redis on localhost:6379.\n"
            "Start Redis with: docker run -d -p 6379:6379 redis:alpine\n"
            "Or: docker-compose up -d redis"
        )
        sys.exit(1)

    # Clear old stream data
    for stream in ["sensor:cmapss", "sensor:ai4i", "predictions"]:
        r.delete(stream)
    logger.info("Cleared old stream data.")

    # Initialize
    producer = StreamProducer(redis_client=r, replay_speed_ms=speed_ms)
    consumer = StreamConsumer(redis_client=r, on_prediction=prediction_callback)

    # Start consumer in background thread
    consumer_thread = threading.Thread(
        target=consumer.consume,
        kwargs={"timeout_ms": 500},
        daemon=True,
    )
    consumer_thread.start()
    logger.info("Consumer thread started.")

    # Give consumer a moment to load models
    time.sleep(2)

    print("\n" + "=" * 70)
    print("MachineGuard Streaming Demo — Live Predictions")
    print("=" * 70 + "\n")

    try:
        if fleet in ("cmapss", "both"):
            producer.replay_cmapss(
                subset="FD001",
                max_engines=max_engines,
            )

        if fleet in ("ai4i", "both"):
            producer.replay_ai4i(max_rows=max_ai4i_rows)

        # Wait for consumer to catch up
        time.sleep(3)

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")

    finally:
        producer.stop()
        consumer.stop()

        print("\n" + "=" * 70)
        print("Demo Summary")
        print("=" * 70)
        stats = consumer.stats
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Active machines tracked: {stats['active_machines']}")
        print(f"  Buffer sizes: {stats['buffer_sizes']}")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="MachineGuard Streaming Demo")
    parser.add_argument(
        "--fleet",
        choices=["cmapss", "ai4i", "both"],
        default="both",
        help="Which fleet to replay",
    )
    parser.add_argument(
        "--speed-ms",
        type=int,
        default=50,
        help="Milliseconds between readings (lower = faster)",
    )
    parser.add_argument(
        "--engines",
        type=int,
        default=3,
        help="Number of C-MAPSS engines to replay",
    )
    parser.add_argument(
        "--ai4i-rows",
        type=int,
        default=200,
        help="Number of AI4I rows to replay",
    )
    args = parser.parse_args()

    run_demo(
        fleet=args.fleet,
        speed_ms=args.speed_ms,
        max_engines=args.engines,
        max_ai4i_rows=args.ai4i_rows,
    )


if __name__ == "__main__":
    main()
