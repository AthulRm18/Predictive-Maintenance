"""RUL (Remaining Useful Life) regression model for C-MAPSS turbofan data.

Uses XGBoost regressor with leave-out-engine validation to prevent data
leakage between cycles of the same engine unit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from machineguard.config import MODELS_DIR, settings
from machineguard.data.cmapss_loader import (
    load_train_test,
    get_sensor_columns,
    load_rul_targets,
)
from machineguard.features.cmapss_features import build_feature_matrix
from machineguard.evaluation.metrics import rul_evaluation_report, asymmetric_rul_score

logger = logging.getLogger(__name__)


class RULModel:
    """XGBoost-based Remaining Useful Life regressor.

    Attributes:
        model: Trained XGBoost regressor.
        feature_names: List of feature column names.
        scaler_params: Normalization parameters from training data.
        metrics: Evaluation metrics from last training run.
    """

    def __init__(self):
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []
        self.scaler_params: dict = {}
        self.metrics: dict[str, Any] = {}

    def train(
        self,
        subset: str = "FD001",
        data_dir: Path | None = None,
        hyperparams: dict | None = None,
        n_eval_engines: int = 20,
    ) -> dict[str, Any]:
        """Train the RUL model with leave-out-engine validation.

        Args:
            subset: C-MAPSS subset to train on.
            data_dir: Override data directory.
            hyperparams: XGBoost hyperparameters override.
            n_eval_engines: Number of engines to hold out for validation.

        Returns:
            Training and validation metrics.
        """
        logger.info(f"Loading C-MAPSS {subset} data...")
        train_df, test_df, true_rul = load_train_test(
            subset=subset,
            normalize=True,
            data_dir=data_dir,
        )

        # Leave-out-engine validation split
        all_units = train_df["unit_id"].unique()
        np.random.seed(42)
        val_units = np.random.choice(all_units, size=n_eval_engines, replace=False)

        train_mask = ~train_df["unit_id"].isin(val_units)
        val_mask = train_df["unit_id"].isin(val_units)

        train_split = train_df[train_mask].copy()
        val_split = train_df[val_mask].copy()

        logger.info(
            f"Train: {len(train_split)} rows ({len(all_units) - n_eval_engines} engines), "
            f"Val: {len(val_split)} rows ({n_eval_engines} engines)"
        )

        # Build features
        logger.info("Building feature matrix...")
        X_train, self.feature_names = build_feature_matrix(train_split)
        X_val, _ = build_feature_matrix(val_split)

        y_train = train_split["rul"].values
        y_val = val_split["rul"].values

        # Default hyperparameters (tuned for C-MAPSS FD001)
        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

        if hyperparams:
            default_params.update(hyperparams)

        logger.info(f"Training XGBoost with params: {default_params}")

        self.model = xgb.XGBRegressor(**default_params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50,
        )

        # Evaluate
        y_val_pred = self.model.predict(X_val)
        y_val_pred = np.clip(y_val_pred, 0, settings.model.rul_cap)

        val_metrics = rul_evaluation_report(y_val, y_val_pred)

        # Also evaluate on test set (last cycle per engine + true RUL)
        test_metrics = self._evaluate_test(test_df, true_rul)

        self.metrics = {
            "validation": val_metrics,
            "test": test_metrics,
            "n_features": len(self.feature_names),
            "hyperparams": default_params,
        }

        logger.info(f"Validation RMSE: {val_metrics['rmse']:.2f}")
        logger.info(f"Validation Asymmetric Score: {val_metrics['asymmetric_score']:.2f}")
        if test_metrics:
            logger.info(f"Test RMSE: {test_metrics['rmse']:.2f}")
            logger.info(f"Test Asymmetric Score: {test_metrics['asymmetric_score']:.2f}")

        return self.metrics

    def _evaluate_test(
        self,
        test_df: pd.DataFrame,
        true_rul: pd.Series,
    ) -> dict[str, Any]:
        """Evaluate on the C-MAPSS test set (last cycle per engine).

        Args:
            test_df: Test DataFrame with truncated trajectories.
            true_rul: True RUL values indexed by unit_id.

        Returns:
            Test evaluation metrics.
        """
        if self.model is None:
            return {}

        # Build features for test data
        X_test, _ = build_feature_matrix(test_df)

        # Get last cycle per engine (this is what we predict RUL for)
        last_cycle_idx = test_df.groupby("unit_id")["cycle"].idxmax()
        X_test_last = X_test.loc[last_cycle_idx]
        test_units = test_df.loc[last_cycle_idx, "unit_id"].values

        # Predict
        y_pred = self.model.predict(X_test_last)
        y_pred = np.clip(y_pred, 0, settings.model.rul_cap)

        # Match with true RUL
        y_true = np.array([true_rul.loc[uid] for uid in test_units])

        return rul_evaluation_report(y_true, y_pred)

    def predict(self, sensor_readings: dict[str, float]) -> dict[str, Any]:
        """Predict RUL for a single reading.

        For single-reading inference (no rolling features), uses only the
        base sensor values. For production use with streaming data, the
        feature engineering layer should provide rolling features.

        Args:
            sensor_readings: Dict of sensor_name → value.

        Returns:
            Dict with predicted RUL and confidence estimate.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first or load a saved model.")

        # Build feature vector
        feature_vector = np.zeros(len(self.feature_names))
        for i, feat_name in enumerate(self.feature_names):
            if feat_name in sensor_readings:
                feature_vector[i] = sensor_readings[feat_name]

        X = pd.DataFrame([feature_vector], columns=self.feature_names)

        # Predict
        rul_pred = float(self.model.predict(X)[0])
        rul_pred = max(0, min(rul_pred, settings.model.rul_cap))

        return {
            "predicted_rul": round(rul_pred, 1),
            "rul_cap": settings.model.rul_cap,
        }

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Predict RUL for a batch of readings.

        Args:
            X: Feature matrix with columns matching self.feature_names.

        Returns:
            Array of predicted RUL values.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        # Ensure correct column order
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        preds = self.model.predict(X_aligned)
        return np.clip(preds, 0, settings.model.rul_cap)

    def save(self, path: Path | None = None) -> Path:
        """Save the trained model and metadata.

        Args:
            path: Override save path.

        Returns:
            Path where model was saved.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        save_dir = path or (MODELS_DIR / "rul_model")
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "model.joblib"
        meta_path = save_dir / "metadata.joblib"

        joblib.dump(self.model, model_path)
        joblib.dump({
            "feature_names": self.feature_names,
            "scaler_params": self.scaler_params,
            "metrics": self.metrics,
        }, meta_path)

        logger.info(f"RUL model saved to {save_dir}")
        return save_dir

    def load(self, path: Path | None = None) -> None:
        """Load a trained model from disk.

        Args:
            path: Override load path.
        """
        load_dir = path or (MODELS_DIR / "rul_model")

        model_path = load_dir / "model.joblib"
        meta_path = load_dir / "metadata.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = joblib.load(model_path)
        meta = joblib.load(meta_path)
        self.feature_names = meta["feature_names"]
        self.scaler_params = meta.get("scaler_params", {})
        self.metrics = meta.get("metrics", {})

        logger.info(f"RUL model loaded from {load_dir}")

    def get_feature_importance(self, top_k: int = 15) -> list[tuple[str, float]]:
        """Get top-k most important features.

        Returns:
            List of (feature_name, importance_score) tuples, sorted descending.
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]

        return [
            (self.feature_names[i], float(importances[i]))
            for i in indices
        ]
