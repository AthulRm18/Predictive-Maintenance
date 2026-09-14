"""Feedback store — records human verdicts and accumulates retraining data.

When a reviewer labels a prediction (confirmed_fault, false_alarm, monitoring),
this store persists the verdict and tracks whether enough labeled data has
accumulated to trigger a retrain cycle.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from machineguard.config import MODELS_DIR

logger = logging.getLogger(__name__)

# Feedback is stored as JSONL files (one per fleet) for simplicity.
# In production with TimescaleDB running, this would query the review_queue table.
FEEDBACK_DIR = MODELS_DIR.parent / "feedback"


class FeedbackStore:
    """Stores and retrieves human feedback on predictions.

    Uses JSONL files as a lightweight feedback store. Each entry contains:
    - The original prediction
    - The sensor readings at prediction time
    - The human verdict (confirmed_fault, false_alarm, monitoring)
    - The reviewer's corrected label (if different from prediction)
    """

    def __init__(self, feedback_dir: Path | None = None):
        self.feedback_dir = feedback_dir or FEEDBACK_DIR
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self._counters: dict[str, int] = {}

    def _get_file(self, fleet_id: str) -> Path:
        return self.feedback_dir / f"feedback_{fleet_id}.jsonl"

    def add_feedback(
        self,
        prediction_id: str | int,
        prediction: dict,
        verdict: str,
        corrected_label: str | None = None,
        reviewer_id: str = "anonymous",
        sensor_readings: dict | None = None,
        notes: str = "",
    ) -> dict:
        """Record a human verdict on a prediction.

        Args:
            prediction_id: Unique ID of the prediction.
            prediction: The original prediction dict.
            verdict: One of: confirmed_fault, false_alarm, monitoring.
            corrected_label: If verdict is correction, the true label.
            reviewer_id: Who reviewed it.
            sensor_readings: The sensor data at prediction time.
            notes: Free-text notes from the reviewer.

        Returns:
            The stored feedback entry.
        """
        valid_verdicts = {"confirmed_fault", "false_alarm", "monitoring", "corrected"}
        if verdict not in valid_verdicts:
            raise ValueError(f"Invalid verdict '{verdict}'. Must be one of: {valid_verdicts}")

        fleet_id = prediction.get("fleet_id", "unknown")
        entry = {
            "prediction_id": str(prediction_id),
            "prediction": prediction,
            "verdict": verdict,
            "corrected_label": corrected_label,
            "reviewer_id": reviewer_id,
            "sensor_readings": sensor_readings,
            "notes": notes,
            "reviewed_at": datetime.utcnow().isoformat(),
        }

        filepath = self._get_file(fleet_id)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        # Update counters
        self._counters[fleet_id] = self._counters.get(fleet_id, 0) + 1

        logger.info(
            f"Feedback recorded: prediction={prediction_id}, "
            f"verdict={verdict}, fleet={fleet_id} "
            f"(total: {self.get_feedback_count(fleet_id)})"
        )

        return entry

    def get_feedback(
        self,
        fleet_id: str,
        verdict_filter: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Retrieve stored feedback entries.

        Args:
            fleet_id: Which fleet's feedback to retrieve.
            verdict_filter: Optional filter by verdict type.
            limit: Max number of entries to return.

        Returns:
            List of feedback entries.
        """
        filepath = self._get_file(fleet_id)
        if not filepath.exists():
            return []

        entries = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if verdict_filter and entry.get("verdict") != verdict_filter:
                    continue
                entries.append(entry)
                if limit and len(entries) >= limit:
                    break

        return entries

    def get_feedback_count(self, fleet_id: str) -> int:
        """Get total feedback count for a fleet."""
        if fleet_id in self._counters:
            return self._counters[fleet_id]

        filepath = self._get_file(fleet_id)
        if not filepath.exists():
            return 0

        count = sum(1 for _ in open(filepath, "r", encoding="utf-8") if _.strip())
        self._counters[fleet_id] = count
        return count

    def get_correction_pairs(self, fleet_id: str) -> list[tuple[dict, str]]:
        """Get (sensor_readings, corrected_label) pairs for retraining.

        Returns only entries where the human corrected the model's prediction.
        These are the most valuable training signals.
        """
        entries = self.get_feedback(fleet_id)
        pairs = []

        for entry in entries:
            corrected = entry.get("corrected_label")
            sensor_data = entry.get("sensor_readings")
            verdict = entry.get("verdict")

            if corrected and sensor_data:
                pairs.append((sensor_data, corrected))
            elif verdict == "confirmed_fault":
                # Model was right — reinforcement signal
                pred = entry.get("prediction", {})
                predicted_class = pred.get("predicted_class")
                if predicted_class and sensor_data:
                    pairs.append((sensor_data, predicted_class))

        return pairs

    def should_trigger_retrain(
        self,
        fleet_id: str,
        min_feedback: int = 20,
        correction_ratio_threshold: float = 0.1,
    ) -> tuple[bool, dict]:
        """Check if enough feedback has accumulated to trigger retraining.

        Triggers when:
        1. At least min_feedback entries exist, AND
        2. The correction ratio exceeds the threshold (model is wrong too often)

        Args:
            fleet_id: Fleet to check.
            min_feedback: Minimum feedback entries needed.
            correction_ratio_threshold: Trigger if correction rate exceeds this.

        Returns:
            Tuple of (should_retrain, stats_dict).
        """
        total = self.get_feedback_count(fleet_id)
        if total < min_feedback:
            return False, {"total": total, "min_required": min_feedback, "reason": "insufficient_data"}

        entries = self.get_feedback(fleet_id)
        verdicts = {}
        for e in entries:
            v = e.get("verdict", "unknown")
            verdicts[v] = verdicts.get(v, 0) + 1

        corrections = verdicts.get("corrected", 0) + verdicts.get("false_alarm", 0)
        correction_ratio = corrections / total if total > 0 else 0

        stats = {
            "total": total,
            "verdicts": verdicts,
            "correction_ratio": round(correction_ratio, 3),
            "threshold": correction_ratio_threshold,
        }

        should_retrain = correction_ratio > correction_ratio_threshold
        if should_retrain:
            stats["reason"] = "correction_ratio_exceeded"
            logger.info(
                f"Retrain triggered for {fleet_id}: "
                f"correction_ratio={correction_ratio:.3f} > {correction_ratio_threshold}"
            )
        else:
            stats["reason"] = "model_performing_well"

        return should_retrain, stats

    def clear(self, fleet_id: str) -> None:
        """Clear feedback for a fleet (after successful retrain)."""
        filepath = self._get_file(fleet_id)
        if filepath.exists():
            filepath.unlink()
        self._counters.pop(fleet_id, None)
        logger.info(f"Cleared feedback for fleet {fleet_id}")


# Singleton
feedback_store = FeedbackStore()
