"""Evaluation metrics for MachineGuard models.

Includes:
- NASA's asymmetric RUL scoring function (penalizes late predictions)
- Per-class recall/precision reporting
- Model comparison for retraining gate (Stage 3)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    recall_score,
    precision_score,
    f1_score,
)


def asymmetric_rul_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA's asymmetric scoring function for RUL prediction.

    Penalizes late predictions (predicting machine is healthy when it's about
    to fail) exponentially more than early predictions.

    Formula:
        For each sample i:
            d_i = predicted_rul - actual_rul
            if d_i < 0 (early prediction):  s_i = exp(-d_i / 13) - 1
            if d_i >= 0 (late prediction):  s_i = exp(d_i / 10) - 1

    Lower score is better. Score = 0 means perfect prediction.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Total asymmetric score (sum over all samples).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    d = y_pred - y_true  # difference

    # Early predictions (d < 0): less penalty
    early_mask = d < 0
    score_early = np.sum(np.exp(-d[early_mask] / 13.0) - 1)

    # Late predictions (d >= 0): more penalty
    late_mask = d >= 0
    score_late = np.sum(np.exp(d[late_mask] / 10.0) - 1)

    return float(score_early + score_late)


def rul_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Comprehensive RUL model evaluation.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Dict with RMSE, MAE, asymmetric score, and distribution stats.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    errors = y_pred - y_true

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(np.mean(np.abs(errors))),
        "asymmetric_score": asymmetric_rul_score(y_true, y_pred),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "pct_early": float(np.mean(errors < 0) * 100),
        "pct_late": float(np.mean(errors >= 0) * 100),
        "max_early_error": float(np.min(errors)) if np.any(errors < 0) else 0.0,
        "max_late_error": float(np.max(errors)) if np.any(errors >= 0) else 0.0,
        "n_samples": len(y_true),
    }


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Per-class recall and precision report for fault classification.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        class_names: Optional list of class names.

    Returns:
        Dict with per-class metrics, macro averages, and confusion matrix.
    """
    # Build labels list: use class_names if provided, else infer from data
    labels = class_names if class_names else sorted(set(y_true) | set(y_pred))

    # Pass both `labels` and `target_names` so sklearn aligns them properly,
    # even when some classes don't appear in y_pred or y_true.
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Extract per-class recall specifically (the key metric per the spec)
    per_class_recall = {}
    for label in labels:
        if label in report:
            per_class_recall[label] = report[label].get("recall", 0.0)

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_class_recall": per_class_recall,
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_samples": len(y_true),
    }


def model_comparison_report(
    old_metrics: dict[str, float],
    new_metrics: dict[str, float],
    metric_key: str = "macro_recall",
    improvement_threshold: float = 0.0,
) -> dict[str, Any]:
    """Compare old and new model metrics for the retraining gate.

    Used in Stage 3: a retrained model is only promoted if it beats the
    current production model on the specified metric.

    Args:
        old_metrics: Metrics from the current production model.
        new_metrics: Metrics from the candidate retrained model.
        metric_key: Which metric to use for comparison.
        improvement_threshold: Minimum improvement required (e.g., 0.01 for 1%).

    Returns:
        Dict with comparison results and promotion recommendation.
    """
    old_val = old_metrics.get(metric_key, 0.0)
    new_val = new_metrics.get(metric_key, 0.0)
    improvement = new_val - old_val

    should_promote = improvement > improvement_threshold

    return {
        "metric": metric_key,
        "old_value": old_val,
        "new_value": new_val,
        "improvement": improvement,
        "improvement_pct": (improvement / max(old_val, 1e-8)) * 100,
        "threshold": improvement_threshold,
        "should_promote": should_promote,
        "reason": (
            f"New model improves {metric_key} by {improvement:.4f} "
            f"({'above' if should_promote else 'below'} threshold {improvement_threshold})"
        ),
    }
