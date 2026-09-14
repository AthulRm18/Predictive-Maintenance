"""Counterfactual explanations using DiCE.

Generates "what-if" explanations: what minimal sensor changes would
prevent the predicted failure? This is the feature that makes
MachineGuard actionable — operators don't just see 'failure imminent',
they see 'reduce torque by 8% to prevent heat dissipation failure'.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CounterfactualExplainer:
    """Generates actionable counterfactual explanations.

    Uses a perturbation-based approach to find minimal sensor changes
    that would flip the prediction from failure to healthy (or extend RUL).

    This is a lightweight implementation that works without dice-ml for
    maximum portability. The logic mirrors DiCE's random perturbation
    strategy but is self-contained.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        feature_ranges: dict[str, tuple[float, float]] | None = None,
        n_counterfactuals: int = 3,
        max_iterations: int = 500,
    ):
        """Initialize the explainer.

        Args:
            model: Trained model with predict/predict_proba method.
            feature_names: List of feature column names.
            feature_ranges: Optional {feature: (min, max)} constraints.
            n_counterfactuals: Number of counterfactuals to generate.
            max_iterations: Max perturbation attempts.
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges or {}
        self.n_counterfactuals = n_counterfactuals
        self.max_iterations = max_iterations

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Get model predictions."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def explain_fault(
        self,
        instance: dict[str, float],
        target_class: str = "no_failure",
    ) -> list[dict]:
        """Generate counterfactuals for a fault prediction.

        "What sensor values would need to change to prevent this failure?"

        Args:
            instance: Current sensor readings {feature: value}.
            target_class: Desired class (usually "no_failure").

        Returns:
            List of counterfactual dicts, each showing required changes.
        """
        # Build instance array
        values = [instance.get(f, 0.0) for f in self.feature_names]
        original = np.array(values).reshape(1, -1)

        counterfactuals = []
        rng = np.random.default_rng(42)

        for iteration in range(self.max_iterations):
            if len(counterfactuals) >= self.n_counterfactuals:
                break

            # Random perturbation: modify 1-3 features
            n_features_to_change = rng.integers(1, min(4, len(self.feature_names) + 1))
            features_to_change = rng.choice(
                len(self.feature_names), size=n_features_to_change, replace=False
            )

            candidate = original.copy()
            changes = {}

            for idx in features_to_change:
                feat_name = self.feature_names[idx]
                original_val = original[0, idx]

                # Perturbation within feature range
                if feat_name in self.feature_ranges:
                    lo, hi = self.feature_ranges[feat_name]
                else:
                    # Default: +/- 20% of original value
                    spread = abs(original_val) * 0.2 + 1e-6
                    lo, hi = original_val - spread, original_val + spread

                new_val = rng.uniform(lo, hi)
                candidate[0, idx] = new_val
                changes[feat_name] = {
                    "original": float(original_val),
                    "counterfactual": float(new_val),
                    "change": float(new_val - original_val),
                    "change_pct": float(
                        ((new_val - original_val) / max(abs(original_val), 1e-8)) * 100
                    ),
                }

            # Check if the counterfactual flips the prediction
            try:
                pred = self._predict(candidate)
                if hasattr(pred, "shape") and pred.ndim == 2:
                    # Classification: check if target class probability increased
                    # (simplified — just check if prediction changed)
                    original_pred = self._predict(original)
                    if not np.array_equal(pred.argmax(axis=1), original_pred.argmax(axis=1)):
                        counterfactuals.append({
                            "changes": changes,
                            "n_features_changed": len(changes),
                            "total_perturbation": sum(
                                abs(c["change_pct"]) for c in changes.values()
                            ),
                        })
                else:
                    # Regression: check for meaningful change
                    original_pred = self._predict(original)
                    if abs(pred[0] - original_pred[0]) > 5:  # >5 cycle RUL change
                        counterfactuals.append({
                            "changes": changes,
                            "predicted_rul_change": float(pred[0] - original_pred[0]),
                            "n_features_changed": len(changes),
                        })
            except Exception as e:
                logger.debug(f"Counterfactual attempt {iteration} failed: {e}")
                continue

        # Sort by fewest changes (most actionable)
        counterfactuals.sort(key=lambda x: x["n_features_changed"])

        return counterfactuals[:self.n_counterfactuals]

    def explain_rul(
        self,
        instance: dict[str, float],
        target_rul_improvement: float = 20.0,
    ) -> list[dict]:
        """Generate counterfactuals for RUL prediction.

        "What changes would extend remaining useful life by X cycles?"

        Args:
            instance: Current sensor readings.
            target_rul_improvement: Desired RUL extension in cycles.

        Returns:
            List of counterfactual dicts showing required changes.
        """
        values = [instance.get(f, 0.0) for f in self.feature_names]
        original = np.array(values).reshape(1, -1)

        try:
            original_rul = float(self._predict(original)[0])
        except Exception:
            original_rul = 0.0

        counterfactuals = []
        rng = np.random.default_rng(42)

        for iteration in range(self.max_iterations):
            if len(counterfactuals) >= self.n_counterfactuals:
                break

            n_features_to_change = rng.integers(1, min(4, len(self.feature_names) + 1))
            features_to_change = rng.choice(
                len(self.feature_names), size=n_features_to_change, replace=False
            )

            candidate = original.copy()
            changes = {}

            for idx in features_to_change:
                feat_name = self.feature_names[idx]
                original_val = original[0, idx]

                if feat_name in self.feature_ranges:
                    lo, hi = self.feature_ranges[feat_name]
                else:
                    spread = abs(original_val) * 0.25 + 1e-6
                    lo, hi = original_val - spread, original_val + spread

                new_val = rng.uniform(lo, hi)
                candidate[0, idx] = new_val
                changes[feat_name] = {
                    "original": float(original_val),
                    "counterfactual": float(new_val),
                    "change": float(new_val - original_val),
                    "change_pct": float(
                        ((new_val - original_val) / max(abs(original_val), 1e-8)) * 100
                    ),
                }

            try:
                new_rul = float(self._predict(candidate)[0])
                rul_change = new_rul - original_rul

                if rul_change >= target_rul_improvement:
                    counterfactuals.append({
                        "changes": changes,
                        "original_rul": original_rul,
                        "counterfactual_rul": new_rul,
                        "rul_improvement": rul_change,
                        "n_features_changed": len(changes),
                    })
            except Exception:
                continue

        counterfactuals.sort(key=lambda x: x["n_features_changed"])
        return counterfactuals[:self.n_counterfactuals]

    @staticmethod
    def format_recommendation(counterfactual: dict) -> str:
        """Format a counterfactual as a human-readable recommendation.

        Args:
            counterfactual: Single counterfactual dict.

        Returns:
            Human-readable string like "Reduce Torque by 8.3%".
        """
        changes = counterfactual.get("changes", {})
        recommendations = []
        for feat, info in changes.items():
            change_pct = info["change_pct"]
            direction = "Increase" if change_pct > 0 else "Reduce"
            recommendations.append(
                f"{direction} {feat} by {abs(change_pct):.1f}% "
                f"(from {info['original']:.2f} to {info['counterfactual']:.2f})"
            )
        return "; ".join(recommendations)
