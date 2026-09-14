"""Tests for evaluation metrics."""

import numpy as np
import pytest

from machineguard.evaluation.metrics import (
    asymmetric_rul_score,
    rul_evaluation_report,
    per_class_report,
    model_comparison_report,
)


class TestAsymmetricRULScore:
    """Test NASA's asymmetric RUL scoring function."""

    def test_perfect_prediction(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])
        assert asymmetric_rul_score(y_true, y_pred) == 0.0

    def test_early_prediction_penalty(self):
        y_true = np.array([50])
        y_pred = np.array([40])  # Early by 10 cycles
        score = asymmetric_rul_score(y_true, y_pred)
        # d = -10, score = exp(10/13) - 1 ≈ 1.154
        assert score > 0
        assert score < 2  # Early penalty is moderate

    def test_late_prediction_penalty(self):
        y_true = np.array([50])
        y_pred = np.array([60])  # Late by 10 cycles
        score = asymmetric_rul_score(y_true, y_pred)
        # d = 10, score = exp(10/10) - 1 ≈ 1.718
        assert score > 0
        assert score > asymmetric_rul_score(np.array([50]), np.array([40]))  # Late > early

    def test_late_is_worse_than_early(self):
        y_true = np.array([50])
        early = asymmetric_rul_score(y_true, np.array([40]))  # 10 early
        late = asymmetric_rul_score(y_true, np.array([60]))   # 10 late
        assert late > early

    def test_batch(self):
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 30, 45, 48])
        score = asymmetric_rul_score(y_true, y_pred)
        assert score > 0


class TestRULEvaluationReport:
    """Test comprehensive RUL evaluation report."""

    def test_report_keys(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 32])
        report = rul_evaluation_report(y_true, y_pred)
        assert "rmse" in report
        assert "mae" in report
        assert "asymmetric_score" in report
        assert "pct_early" in report
        assert "pct_late" in report

    def test_report_values(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])
        report = rul_evaluation_report(y_true, y_pred)
        assert report["rmse"] == 0.0
        assert report["mae"] == 0.0
        assert report["asymmetric_score"] == 0.0


class TestPerClassReport:
    """Test fault classification metrics."""

    def test_perfect_classification(self):
        y_true = ["A", "B", "C", "A", "B"]
        y_pred = ["A", "B", "C", "A", "B"]
        report = per_class_report(y_true, y_pred, ["A", "B", "C"])
        assert report["macro_recall"] == 1.0
        assert report["macro_f1"] == 1.0

    def test_report_structure(self):
        y_true = ["no_failure", "no_failure", "HDF", "no_failure"]
        y_pred = ["no_failure", "HDF", "HDF", "no_failure"]
        report = per_class_report(y_true, y_pred, ["no_failure", "HDF"])
        assert "per_class_recall" in report
        assert "confusion_matrix" in report
        assert "macro_f1" in report


class TestModelComparison:
    """Test model comparison for retraining gate."""

    def test_should_promote(self):
        old = {"macro_recall": 0.80}
        new = {"macro_recall": 0.85}
        result = model_comparison_report(old, new)
        assert result["should_promote"] is True
        assert result["improvement"] == pytest.approx(0.05)

    def test_should_not_promote(self):
        old = {"macro_recall": 0.85}
        new = {"macro_recall": 0.83}
        result = model_comparison_report(old, new)
        assert result["should_promote"] is False

    def test_threshold(self):
        old = {"macro_recall": 0.80}
        new = {"macro_recall": 0.805}
        result = model_comparison_report(old, new, improvement_threshold=0.01)
        assert result["should_promote"] is False
