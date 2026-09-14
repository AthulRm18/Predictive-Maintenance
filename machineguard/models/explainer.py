"""SHAP explainability wrapper for MachineGuard models.

Provides unified SHAP explanation interface for both the RUL regressor
and the fault classifier. Uses TreeExplainer for exact, fast SHAP values
on XGBoost models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

from machineguard.config import settings

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Unified SHAP explainability wrapper.

    Caches the SHAP explainer object (expensive to create) and provides
    methods for single-instance and batch explanations.

    Attributes:
        explainer: Cached SHAP TreeExplainer.
        feature_names: Feature column names.
        model_type: "regressor" or "classifier".
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        model_type: str = "regressor",
        class_names: list[str] | None = None,
    ):
        """Initialize the explainer.

        Args:
            model: Trained XGBoost model (XGBRegressor or XGBClassifier).
            feature_names: List of feature column names.
            model_type: "regressor" for RUL, "classifier" for fault classification.
            class_names: Class names for classifier (required if model_type="classifier").
        """
        self.feature_names = feature_names
        self.model_type = model_type
        self.class_names = class_names or []
        self.top_k = settings.model.shap_top_k

        logger.info(f"Creating SHAP TreeExplainer for {model_type}...")
        self.explainer = shap.TreeExplainer(model)
        logger.info("SHAP explainer ready.")

    def explain_single(
        self,
        sensor_readings: dict[str, float],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate SHAP explanation for a single prediction.

        Args:
            sensor_readings: Dict of feature_name → value.
            top_k: Number of top features to return.

        Returns:
            Dict with SHAP values, top features, and force plot data.
        """
        if top_k is None:
            top_k = self.top_k

        # Build feature vector
        feature_vector = np.zeros(len(self.feature_names))
        for i, feat_name in enumerate(self.feature_names):
            if feat_name in sensor_readings:
                feature_vector[i] = sensor_readings[feat_name]

        X = pd.DataFrame([feature_vector], columns=self.feature_names)

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)

        if self.model_type == "classifier":
            return self._format_classifier_explanation(
                shap_values, feature_vector, top_k
            )
        else:
            return self._format_regressor_explanation(
                shap_values, feature_vector, top_k
            )

    def explain_batch(
        self,
        X: pd.DataFrame,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate SHAP explanations for a batch of predictions.

        Args:
            X: Feature matrix.
            top_k: Number of top features per explanation.

        Returns:
            List of explanation dicts.
        """
        if top_k is None:
            top_k = self.top_k

        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        shap_values = self.explainer.shap_values(X_aligned)

        explanations = []
        for i in range(len(X_aligned)):
            if self.model_type == "classifier":
                exp = self._format_classifier_explanation(
                    [sv[i:i+1] for sv in shap_values] if isinstance(shap_values, list) else shap_values[i:i+1],
                    X_aligned.iloc[i].values,
                    top_k,
                )
            else:
                row_sv = shap_values[i] if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2 else shap_values[i]
                exp = self._format_regressor_explanation(
                    row_sv.reshape(1, -1) if hasattr(row_sv, 'reshape') else np.array([row_sv]),
                    X_aligned.iloc[i].values,
                    top_k,
                )
            explanations.append(exp)

        return explanations

    def _format_regressor_explanation(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        top_k: int,
    ) -> dict[str, Any]:
        """Format SHAP explanation for regressor output."""
        # shap_values shape: (1, n_features) for single instance
        sv = np.asarray(shap_values).flatten()

        # All SHAP values
        all_shap = {
            name: round(float(sv[i]), 4)
            for i, name in enumerate(self.feature_names)
        }

        # Top-k by absolute value
        abs_indices = np.argsort(np.abs(sv))[::-1][:top_k]
        top_features = [
            {
                "feature": self.feature_names[idx],
                "shap_value": round(float(sv[idx]), 4),
                "feature_value": round(float(feature_values[idx]), 4),
                "direction": "increases RUL" if sv[idx] > 0 else "decreases RUL",
            }
            for idx in abs_indices
        ]

        # Base value (expected value)
        base_value = float(self.explainer.expected_value)
        if isinstance(self.explainer.expected_value, np.ndarray):
            base_value = float(self.explainer.expected_value[0])

        return {
            "shap_values": all_shap,
            "top_features": top_features,
            "base_value": round(base_value, 4),
            "model_type": "regressor",
        }

    def _format_classifier_explanation(
        self,
        shap_values: Any,
        feature_values: np.ndarray,
        top_k: int,
    ) -> dict[str, Any]:
        """Format SHAP explanation for classifier output.

        For multi-class classifiers, SHAP returns values per class.
        We report the explanation for the predicted class.
        """
        if isinstance(shap_values, list):
            # List of arrays, one per class
            # Find predicted class (class with highest sum of SHAP values + base)
            class_contributions = [np.sum(sv) for sv in shap_values]
            pred_class_idx = int(np.argmax(class_contributions))
            sv = np.asarray(shap_values[pred_class_idx]).flatten()

            per_class_shap = {}
            for cls_idx, cls_name in enumerate(self.class_names):
                cls_sv = np.asarray(shap_values[cls_idx]).flatten()
                abs_indices = np.argsort(np.abs(cls_sv))[::-1][:5]
                per_class_shap[cls_name] = [
                    {
                        "feature": self.feature_names[idx],
                        "shap_value": round(float(cls_sv[idx]), 4),
                    }
                    for idx in abs_indices
                ]
        else:
            sv = np.asarray(shap_values).flatten()
            pred_class_idx = 0
            per_class_shap = {}

        # Top features for predicted class
        abs_indices = np.argsort(np.abs(sv))[::-1][:top_k]
        top_features = [
            {
                "feature": self.feature_names[idx],
                "shap_value": round(float(sv[idx]), 4),
                "feature_value": round(float(feature_values[idx]), 4),
            }
            for idx in abs_indices
        ]

        # All SHAP values for predicted class
        all_shap = {
            name: round(float(sv[i]), 4)
            for i, name in enumerate(self.feature_names)
        }

        # Base values
        base_values = self.explainer.expected_value
        if isinstance(base_values, np.ndarray):
            base_value = float(base_values[pred_class_idx])
        elif isinstance(base_values, list):
            base_value = float(base_values[pred_class_idx]) if pred_class_idx < len(base_values) else 0.0
        else:
            base_value = float(base_values)

        pred_class_name = (
            self.class_names[pred_class_idx]
            if pred_class_idx < len(self.class_names)
            else f"class_{pred_class_idx}"
        )

        return {
            "shap_values": all_shap,
            "top_features": top_features,
            "predicted_class": pred_class_name,
            "base_value": round(base_value, 4),
            "per_class_top_features": per_class_shap,
            "model_type": "classifier",
        }

    def get_global_importance(self, X: pd.DataFrame, top_k: int = 15) -> list[dict[str, Any]]:
        """Compute global feature importance using mean |SHAP|.

        Args:
            X: Feature matrix (representative sample).
            top_k: Number of top features.

        Returns:
            List of feature importance dicts.
        """
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        shap_values = self.explainer.shap_values(X_aligned)

        if isinstance(shap_values, list):
            # Multi-class: average across classes
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        indices = np.argsort(mean_abs)[::-1][:top_k]

        return [
            {
                "feature": self.feature_names[idx],
                "mean_abs_shap": round(float(mean_abs[idx]), 4),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(indices)
        ]
