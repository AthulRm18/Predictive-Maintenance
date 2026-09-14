"""HITL routing logic — decides which predictions need human review.

Most student predictive-maintenance projects stop at 'model shows a red
light.' A real system routes uncertain or high-stakes predictions to a
human and learns from their verdict.

Routing criteria:
1. Uncertain predictions (confidence between low/high thresholds)
2. Model disagreement (RUL says safe, fault says failure, or vice versa)
3. Rare/novel failure modes (first few predictions of an unusual class)
4. Anomaly score above threshold (even if classified as no_failure)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from machineguard.config import settings

logger = logging.getLogger(__name__)


class ReviewReason(str, Enum):
    """Why a prediction was routed to human review."""
    UNCERTAIN_CONFIDENCE = "uncertain_confidence"
    HIGH_ANOMALY_SCORE = "high_anomaly_score"
    RARE_FAILURE_MODE = "rare_failure_mode"
    MODEL_DISAGREEMENT = "model_disagreement"
    CRITICAL_RUL = "critical_rul"
    MANUAL_FLAG = "manual_flag"


@dataclass
class RoutingDecision:
    """Result of the routing check."""
    needs_review: bool
    reasons: list[ReviewReason]
    priority: str  # "low", "medium", "high", "critical"
    auto_alert: bool  # Whether to also fire an automated alert


class ReviewRouter:
    """Routes predictions to the HITL review queue based on configurable rules.

    The router examines each prediction and decides whether it should be:
    - Auto-accepted (high confidence, no anomaly)
    - Routed to human review (uncertain, anomalous, or rare)
    - Auto-alerted (very high confidence of imminent failure)
    """

    def __init__(
        self,
        confidence_low: float | None = None,
        confidence_high: float | None = None,
        anomaly_threshold: float = 0.7,
        critical_rul_threshold: int = 15,
    ):
        self.confidence_low = confidence_low or settings.model.uncertainty_low
        self.confidence_high = confidence_high or settings.model.uncertainty_high
        self.anomaly_threshold = anomaly_threshold
        self.critical_rul_threshold = critical_rul_threshold

        # Track rare failure modes seen count
        self._mode_counts: dict[str, int] = {}
        self._rare_threshold = 5  # First N predictions of a mode go to review

    def check_fault_prediction(self, prediction: dict) -> RoutingDecision:
        """Check if a fault prediction needs human review.

        Args:
            prediction: Fault prediction dict with keys like
                predicted_class, confidence, anomaly_score, probabilities.

        Returns:
            RoutingDecision with review status and reasons.
        """
        reasons: list[ReviewReason] = []
        priority = "low"

        confidence = prediction.get("confidence", 0.0)
        anomaly_score = prediction.get("anomaly_score", 0.0)
        predicted_class = prediction.get("predicted_class", "no_failure")

        # Rule 1: Uncertain confidence
        if self.confidence_low < confidence < self.confidence_high:
            reasons.append(ReviewReason.UNCERTAIN_CONFIDENCE)
            priority = "medium"

        # Rule 2: High anomaly score
        if anomaly_score > self.anomaly_threshold:
            reasons.append(ReviewReason.HIGH_ANOMALY_SCORE)
            priority = max(priority, "high", key=lambda p: _PRIORITY_ORDER.get(p, 0))

        # Rule 3: Rare failure mode
        if predicted_class != "no_failure":
            self._mode_counts[predicted_class] = self._mode_counts.get(predicted_class, 0) + 1
            if self._mode_counts[predicted_class] <= self._rare_threshold:
                reasons.append(ReviewReason.RARE_FAILURE_MODE)
                priority = max(priority, "medium", key=lambda p: _PRIORITY_ORDER.get(p, 0))

        needs_review = len(reasons) > 0
        auto_alert = anomaly_score > 0.9 and confidence > self.confidence_high

        return RoutingDecision(
            needs_review=needs_review,
            reasons=reasons,
            priority=priority,
            auto_alert=auto_alert,
        )

    def check_rul_prediction(self, prediction: dict) -> RoutingDecision:
        """Check if a RUL prediction needs human review.

        Args:
            prediction: RUL prediction dict with predicted_rul, cycle, etc.

        Returns:
            RoutingDecision with review status and reasons.
        """
        reasons: list[ReviewReason] = []
        priority = "low"

        predicted_rul = prediction.get("predicted_rul", 999)

        # Rule: Critical RUL (imminent failure zone)
        if predicted_rul <= self.critical_rul_threshold:
            reasons.append(ReviewReason.CRITICAL_RUL)
            priority = "critical"

        needs_review = len(reasons) > 0
        auto_alert = predicted_rul <= 5  # Very imminent

        return RoutingDecision(
            needs_review=needs_review,
            reasons=reasons,
            priority=priority,
            auto_alert=auto_alert,
        )

    def check_model_disagreement(
        self,
        rul_prediction: dict | None,
        fault_prediction: dict | None,
    ) -> RoutingDecision | None:
        """Check if RUL and fault models disagree.

        Disagreement: RUL says healthy (high RUL) but fault says failure,
        or vice versa.
        """
        if not rul_prediction or not fault_prediction:
            return None

        predicted_rul = rul_prediction.get("predicted_rul", 999)
        predicted_class = fault_prediction.get("predicted_class", "no_failure")

        # RUL says healthy but fault says failure
        rul_healthy = predicted_rul > 50
        fault_failure = predicted_class != "no_failure"

        if rul_healthy and fault_failure:
            return RoutingDecision(
                needs_review=True,
                reasons=[ReviewReason.MODEL_DISAGREEMENT],
                priority="high",
                auto_alert=False,
            )

        # RUL says imminent failure but fault says no failure
        rul_critical = predicted_rul <= self.critical_rul_threshold
        if rul_critical and not fault_failure:
            return RoutingDecision(
                needs_review=True,
                reasons=[ReviewReason.MODEL_DISAGREEMENT],
                priority="high",
                auto_alert=False,
            )

        return None


# Priority ordering for max() comparisons
_PRIORITY_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}
