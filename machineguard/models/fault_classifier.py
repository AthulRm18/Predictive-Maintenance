"""Multi-class fault classifier for AI4I 2020 Predictive Maintenance data.

XGBoost classifier with class imbalance handling (SMOTE + class weights)
and per-class recall as the primary evaluation metric.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from machineguard.config import MODELS_DIR, settings
from machineguard.data.ai4i_loader import load_prepared, FAILURE_MODES
from machineguard.evaluation.metrics import per_class_report

logger = logging.getLogger(__name__)


class FaultClassifier:
    """XGBoost-based multi-class fault classifier.

    Handles the severe class imbalance in AI4I (97%+ no_failure) using:
    - Class-weighted loss function
    - Optional SMOTE oversampling
    - Stratified cross-validation
    - Per-class recall as the primary metric

    Attributes:
        model: Trained XGBoost classifier.
        label_encoder: Maps class names ↔ integer labels.
        feature_names: List of feature column names.
        class_names: Ordered list of class names.
        metrics: Evaluation metrics from last training run.
    """

    def __init__(self):
        self.model: xgb.XGBClassifier | None = None
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.feature_names: list[str] = []
        self.class_names: list[str] = []
        self.metrics: dict[str, Any] = {}

    def train(
        self,
        data_dir: Path | None = None,
        use_smote: bool = True,
        hyperparams: dict | None = None,
    ) -> dict[str, Any]:
        """Train the fault classifier.

        Args:
            data_dir: Override data directory.
            use_smote: Whether to apply SMOTE oversampling for minority classes.
            hyperparams: XGBoost hyperparameters override.

        Returns:
            Training and evaluation metrics.
        """
        logger.info("Loading AI4I 2020 data...")
        data = load_prepared(data_dir=data_dir)

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train_multi"]
        y_test = data["y_test_multi"]

        self.feature_names = data["feature_names"]
        self.class_names = data["class_names"]

        # Encode labels to integers
        self.label_encoder.fit(self.class_names)
        y_train_enc = self.label_encoder.transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)

        logger.info(f"Class distribution (train):")
        for cls in self.class_names:
            count = (y_train == cls).sum()
            logger.info(f"  {cls}: {count} ({count / len(y_train) * 100:.1f}%)")

        # Compute class weights (inverse frequency)
        class_counts = np.bincount(y_train_enc, minlength=len(self.class_names))
        total = len(y_train_enc)
        sample_weights = np.array([
            total / (len(self.class_names) * max(class_counts[label], 1))
            for label in y_train_enc
        ])

        # SMOTE oversampling
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE

                # SMOTE requires minimum k_neighbors samples per class
                min_samples = min(class_counts[class_counts > 0])
                k_neighbors = min(5, max(1, min_samples - 1))

                if min_samples >= 2:
                    logger.info(f"Applying SMOTE (k_neighbors={k_neighbors})...")
                    smote = SMOTE(
                        random_state=42,
                        k_neighbors=k_neighbors,
                    )
                    X_train_resampled, y_train_resampled = smote.fit_resample(
                        X_train, y_train_enc
                    )
                    # Recompute sample weights for resampled data
                    resampled_counts = np.bincount(y_train_resampled)
                    resampled_total = len(y_train_resampled)
                    sample_weights = np.array([
                        resampled_total / (len(self.class_names) * max(resampled_counts[label], 1))
                        for label in y_train_resampled
                    ])
                    logger.info(f"After SMOTE: {len(X_train_resampled)} samples")
                else:
                    logger.warning(
                        f"Skipping SMOTE: minimum class has only {min_samples} samples. "
                        "Using class weights only."
                    )
                    X_train_resampled = X_train.values if hasattr(X_train, 'values') else X_train
                    y_train_resampled = y_train_enc

            except ImportError:
                logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
                X_train_resampled = X_train.values if hasattr(X_train, 'values') else X_train
                y_train_resampled = y_train_enc
        else:
            X_train_resampled = X_train.values if hasattr(X_train, 'values') else X_train
            y_train_resampled = y_train_enc

        # Default hyperparameters
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "num_class": len(self.class_names),
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }

        if hyperparams:
            default_params.update(hyperparams)

        logger.info(f"Training XGBoost classifier...")

        self.model = xgb.XGBClassifier(**default_params)

        self.model.fit(
            X_train_resampled,
            y_train_resampled,
            sample_weight=sample_weights,
            eval_set=[(X_test.values if hasattr(X_test, 'values') else X_test, y_test_enc)],
            verbose=50,
        )

        # Evaluate
        y_pred_enc = self.model.predict(X_test.values if hasattr(X_test, 'values') else X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_enc)

        test_metrics = per_class_report(y_test.values, y_pred, self.class_names)

        self.metrics = {
            "test": test_metrics,
            "n_features": len(self.feature_names),
            "n_classes": len(self.class_names),
            "class_names": self.class_names,
            "used_smote": use_smote,
            "hyperparams": default_params,
        }

        logger.info(f"Test Macro Recall: {test_metrics['macro_recall']:.3f}")
        logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.3f}")
        logger.info(f"Per-class recall: {test_metrics['per_class_recall']}")

        return self.metrics

    def predict(self, sensor_readings: dict[str, float]) -> dict[str, Any]:
        """Predict failure mode for a single reading.

        Args:
            sensor_readings: Dict of feature_name → value.

        Returns:
            Dict with predicted class, probabilities, and anomaly score.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first or load a saved model.")

        # Build feature vector
        feature_vector = np.zeros(len(self.feature_names))
        for i, feat_name in enumerate(self.feature_names):
            if feat_name in sensor_readings:
                feature_vector[i] = sensor_readings[feat_name]

        X = np.array([feature_vector])

        # Get probabilities
        probas = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probas))
        pred_class = self.label_encoder.inverse_transform([pred_idx])[0]

        # Anomaly score: 1 - P(no_failure)
        no_failure_idx = list(self.label_encoder.classes_).index("no_failure")
        anomaly_score = 1.0 - probas[no_failure_idx]

        # Confidence: max probability
        confidence = float(np.max(probas))

        return {
            "predicted_class": pred_class,
            "probabilities": {
                cls: float(probas[i])
                for i, cls in enumerate(self.label_encoder.classes_)
            },
            "anomaly_score": round(anomaly_score, 4),
            "confidence": round(confidence, 4),
        }

    def predict_batch(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict failure modes for a batch.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (predicted_classes, probability_matrix).
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        X_vals = X_aligned.values if hasattr(X_aligned, 'values') else X_aligned

        probas = self.model.predict_proba(X_vals)
        pred_indices = np.argmax(probas, axis=1)
        pred_classes = self.label_encoder.inverse_transform(pred_indices)

        return pred_classes, probas

    def save(self, path: Path | None = None) -> Path:
        """Save the trained model and metadata."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        save_dir = path or (MODELS_DIR / "fault_classifier")
        save_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, save_dir / "model.joblib")
        joblib.dump({
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "label_encoder": self.label_encoder,
            "metrics": self.metrics,
        }, save_dir / "metadata.joblib")

        logger.info(f"Fault classifier saved to {save_dir}")
        return save_dir

    def load(self, path: Path | None = None) -> None:
        """Load a trained model from disk."""
        load_dir = path or (MODELS_DIR / "fault_classifier")

        model_path = load_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = joblib.load(model_path)
        meta = joblib.load(load_dir / "metadata.joblib")
        self.feature_names = meta["feature_names"]
        self.class_names = meta["class_names"]
        self.label_encoder = meta["label_encoder"]
        self.metrics = meta.get("metrics", {})

        logger.info(f"Fault classifier loaded from {load_dir}")

    def get_feature_importance(self, top_k: int = 15) -> list[tuple[str, float]]:
        """Get top-k most important features."""
        if self.model is None:
            raise RuntimeError("No model loaded.")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]

        return [
            (self.feature_names[i], float(importances[i]))
            for i in indices
        ]
