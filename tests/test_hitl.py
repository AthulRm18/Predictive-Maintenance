"""Tests for HITL (Human-in-the-Loop) components.

Tests the review router, feedback store, model gate, and API endpoints.
All tests run without external dependencies (no Redis, no DB).
"""

import json
import tempfile
from pathlib import Path

import pytest

from machineguard.hitl.router import ReviewRouter, ReviewReason


class TestReviewRouter:
    """Test the HITL routing logic."""

    def setup_method(self):
        self.router = ReviewRouter(
            confidence_low=0.3,
            confidence_high=0.7,
            anomaly_threshold=0.7,
            critical_rul_threshold=15,
        )

    def test_high_confidence_no_review(self):
        """High-confidence predictions should not need review."""
        prediction = {
            "type": "fault",
            "predicted_class": "no_failure",
            "confidence": 0.95,
            "anomaly_score": 0.05,
        }
        decision = self.router.check_fault_prediction(prediction)
        assert not decision.needs_review

    def test_uncertain_confidence_needs_review(self):
        """Uncertain predictions should be routed to review."""
        prediction = {
            "type": "fault",
            "predicted_class": "HDF",
            "confidence": 0.5,  # Between 0.3 and 0.7
            "anomaly_score": 0.3,
        }
        decision = self.router.check_fault_prediction(prediction)
        assert decision.needs_review
        assert ReviewReason.UNCERTAIN_CONFIDENCE in decision.reasons

    def test_high_anomaly_needs_review(self):
        """High anomaly score should trigger review."""
        prediction = {
            "type": "fault",
            "predicted_class": "no_failure",
            "confidence": 0.9,
            "anomaly_score": 0.85,
        }
        decision = self.router.check_fault_prediction(prediction)
        assert decision.needs_review
        assert ReviewReason.HIGH_ANOMALY_SCORE in decision.reasons
        assert decision.priority == "high"

    def test_rare_failure_mode(self):
        """First predictions of a rare mode should be reviewed."""
        prediction = {
            "type": "fault",
            "predicted_class": "TWF",
            "confidence": 0.8,
            "anomaly_score": 0.4,
        }
        decision = self.router.check_fault_prediction(prediction)
        assert decision.needs_review
        assert ReviewReason.RARE_FAILURE_MODE in decision.reasons

    def test_critical_rul(self):
        """Low RUL predictions should be routed to review."""
        prediction = {
            "type": "rul",
            "predicted_rul": 10,
        }
        decision = self.router.check_rul_prediction(prediction)
        assert decision.needs_review
        assert ReviewReason.CRITICAL_RUL in decision.reasons
        assert decision.priority == "critical"

    def test_safe_rul_no_review(self):
        """Safe RUL predictions should not need review."""
        prediction = {
            "type": "rul",
            "predicted_rul": 100,
        }
        decision = self.router.check_rul_prediction(prediction)
        assert not decision.needs_review

    def test_model_disagreement(self):
        """RUL-healthy + fault-failure should trigger disagreement."""
        rul = {"predicted_rul": 80}
        fault = {"predicted_class": "HDF"}
        decision = self.router.check_model_disagreement(rul, fault)
        assert decision is not None
        assert decision.needs_review
        assert ReviewReason.MODEL_DISAGREEMENT in decision.reasons

    def test_no_disagreement(self):
        """Consistent predictions should not trigger disagreement."""
        rul = {"predicted_rul": 80}
        fault = {"predicted_class": "no_failure"}
        decision = self.router.check_model_disagreement(rul, fault)
        assert decision is None


class TestFeedbackStore:
    """Test the feedback store."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        from machineguard.hitl.feedback_store import FeedbackStore
        self.store = FeedbackStore(feedback_dir=Path(self.tmpdir))

    def test_add_and_retrieve_feedback(self):
        """Test recording and retrieving feedback."""
        self.store.add_feedback(
            prediction_id="pred-001",
            prediction={"fleet_id": "cmapss", "type": "rul", "predicted_rul": 20},
            verdict="confirmed_fault",
            reviewer_id="test_user",
        )

        entries = self.store.get_feedback("cmapss")
        assert len(entries) == 1
        assert entries[0]["verdict"] == "confirmed_fault"
        assert entries[0]["reviewer_id"] == "test_user"

    def test_feedback_count(self):
        """Test feedback counting."""
        for i in range(5):
            self.store.add_feedback(
                prediction_id=f"pred-{i}",
                prediction={"fleet_id": "ai4i", "type": "fault"},
                verdict="confirmed_fault",
            )

        assert self.store.get_feedback_count("ai4i") == 5
        assert self.store.get_feedback_count("cmapss") == 0

    def test_invalid_verdict_raises(self):
        """Invalid verdicts should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid verdict"):
            self.store.add_feedback(
                prediction_id="pred-001",
                prediction={"fleet_id": "cmapss"},
                verdict="invalid_verdict",
            )

    def test_retrain_trigger_insufficient_data(self):
        """Should not trigger retrain with too few entries."""
        self.store.add_feedback(
            prediction_id="pred-001",
            prediction={"fleet_id": "cmapss"},
            verdict="confirmed_fault",
        )
        should_retrain, stats = self.store.should_trigger_retrain("cmapss")
        assert not should_retrain
        assert stats["reason"] == "insufficient_data"

    def test_retrain_trigger_high_correction_rate(self):
        """Should trigger retrain when correction rate is high."""
        # 15 confirmed, 10 false alarms = 40% correction rate
        for i in range(15):
            self.store.add_feedback(
                prediction_id=f"pred-{i}",
                prediction={"fleet_id": "ai4i"},
                verdict="confirmed_fault",
            )
        for i in range(10):
            self.store.add_feedback(
                prediction_id=f"false-{i}",
                prediction={"fleet_id": "ai4i"},
                verdict="false_alarm",
            )

        should_retrain, stats = self.store.should_trigger_retrain("ai4i", min_feedback=20)
        assert should_retrain
        assert stats["correction_ratio"] > 0.1

    def test_clear_feedback(self):
        """Test clearing feedback."""
        self.store.add_feedback(
            prediction_id="pred-001",
            prediction={"fleet_id": "cmapss"},
            verdict="confirmed_fault",
        )
        self.store.clear("cmapss")
        assert self.store.get_feedback_count("cmapss") == 0


class TestModelGate:
    """Test the model promotion gate."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        from machineguard.hitl.retrain_gate import ModelGate
        self.gate = ModelGate(models_dir=Path(self.tmpdir))
        # Ensure clean registry (override parent-based path to stay in tmpdir)
        self.gate.registry_path = Path(self.tmpdir) / "model_registry.json"

    def test_first_model_auto_promotes(self):
        """First model should be auto-promoted (no production baseline)."""
        result = self.gate.evaluate_and_promote(
            model_name="rul_xgboost",
            candidate_metrics={"rmse": 20.0, "asymmetric_score": 740},
            candidate_version="v0.1.0",
        )
        assert result["promoted"]
        assert result["reason"] == "no_existing_production_model"

    def test_better_model_promotes(self):
        """Better candidate should be promoted."""
        # Set up production model
        self.gate._promote("rul_xgboost", "v0.1.0", {"rmse": 25.0})

        # Candidate with lower RMSE (better)
        result = self.gate.evaluate_and_promote(
            model_name="rul_xgboost",
            candidate_metrics={"rmse": 18.0},
            candidate_version="v0.2.0",
        )
        assert result["promoted"]

    def test_worse_model_blocked(self):
        """Worse candidate should not be promoted."""
        self.gate._promote("rul_xgboost", "v0.1.0", {"rmse": 18.0})

        result = self.gate.evaluate_and_promote(
            model_name="rul_xgboost",
            candidate_metrics={"rmse": 25.0},
            candidate_version="v0.2.0",
        )
        assert not result["promoted"]
        assert result["reason"] == "insufficient_improvement"

    def test_registry_status(self):
        """Test registry status reporting."""
        self.gate._promote("rul_xgboost", "v0.1.0", {"rmse": 20.0})
        status = self.gate.get_registry_status()
        assert "rul_xgboost" in status["models"]
        assert status["models"]["rul_xgboost"]["status"] == "production"


class TestHITLAPI:
    """Test the HITL API endpoints."""

    def setup_method(self):
        from fastapi.testclient import TestClient
        from machineguard.api.app import create_app
        app = create_app()
        self.client = TestClient(app)

    def test_route_prediction(self):
        """Test routing a prediction."""
        response = self.client.post(
            "/api/v1/review/route",
            json={
                "type": "fault",
                "predicted_class": "HDF",
                "confidence": 0.5,
                "anomaly_score": 0.3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "needs_review" in data

    def test_get_empty_review_queue(self):
        """Empty queue should return empty list."""
        response = self.client.get("/api/v1/review/queue")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_feedback_stats(self):
        """Test feedback stats endpoint."""
        response = self.client.get("/api/v1/review/stats/cmapss")
        assert response.status_code == 200
        data = response.json()
        assert "fleet_id" in data
        assert "should_retrain" in data

    def test_get_model_registry(self):
        """Test model registry endpoint."""
        response = self.client.get("/api/v1/models/registry")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "history_count" in data
