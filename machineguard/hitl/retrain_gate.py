"""Retrain trigger and MLflow model promotion gate.

Orchestrates the retrain → evaluate → compare → promote pipeline.
A new model is only promoted to production if it beats the current
champion on the key metrics (RMSE for RUL, macro recall for fault).

This is the "closed loop" that turns HITL from a label-collection exercise
into an actual model improvement system.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from machineguard.config import MODELS_DIR, settings
from machineguard.evaluation.metrics import model_comparison_report

logger = logging.getLogger(__name__)


class ModelGate:
    """Compares candidate models against production and gates promotion.

    The gate loads metrics from the training summary, compares them
    against the current production model's metrics, and only promotes
    if the candidate is meaningfully better.
    """

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or MODELS_DIR
        self.registry_path = self.models_dir.parent / "model_registry.json"

    def _load_registry(self) -> dict:
        """Load the local model registry."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"models": {}, "history": []}

    def _save_registry(self, registry: dict) -> None:
        """Save the local model registry."""
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2, default=str)

    def get_production_metrics(self, model_name: str) -> dict | None:
        """Get metrics for the current production model."""
        registry = self._load_registry()
        model_info = registry.get("models", {}).get(model_name)
        if model_info and model_info.get("status") == "production":
            return model_info.get("metrics")
        return None

    def register_candidate(
        self,
        model_name: str,
        version: str,
        metrics: dict,
        artifact_path: str | None = None,
    ) -> dict:
        """Register a candidate model in the local registry.

        Args:
            model_name: e.g., "rul_xgboost" or "fault_xgboost"
            version: Version string (e.g., "v0.2.0" or timestamp)
            metrics: Training/test metrics dict
            artifact_path: Path to model artifacts

        Returns:
            Registry entry.
        """
        registry = self._load_registry()
        entry = {
            "model_name": model_name,
            "version": version,
            "status": "candidate",
            "metrics": metrics,
            "artifact_path": artifact_path or str(self.models_dir / model_name),
            "created_at": datetime.utcnow().isoformat(),
        }

        registry["history"].append(entry)
        logger.info(f"Registered candidate: {model_name} {version}")

        self._save_registry(registry)
        return entry

    def evaluate_and_promote(
        self,
        model_name: str,
        candidate_metrics: dict,
        candidate_version: str,
        improvement_threshold: float = 0.02,
    ) -> dict:
        """Compare candidate against production and promote if better.

        Args:
            model_name: Model identifier.
            candidate_metrics: Metrics from the candidate model.
            candidate_version: Version string for the candidate.
            improvement_threshold: Minimum relative improvement required.

        Returns:
            Dict with decision details: promoted, comparison, reason.
        """
        production_metrics = self.get_production_metrics(model_name)

        if production_metrics is None:
            # No production model exists — auto-promote
            self._promote(model_name, candidate_version, candidate_metrics)
            return {
                "promoted": True,
                "reason": "no_existing_production_model",
                "version": candidate_version,
                "metrics": candidate_metrics,
            }

        # Compare using the model comparison report
        comparison = model_comparison_report(
            production_metrics,
            candidate_metrics,
            improvement_threshold=improvement_threshold,
        )

        if comparison["should_promote"]:
            self._promote(model_name, candidate_version, candidate_metrics)
            return {
                "promoted": True,
                "reason": "candidate_outperforms_production",
                "version": candidate_version,
                "comparison": comparison,
            }
        else:
            logger.info(
                f"Model gate: {model_name} {candidate_version} NOT promoted. "
                f"Improvement insufficient: {comparison.get('details', {})}"
            )
            return {
                "promoted": False,
                "reason": "insufficient_improvement",
                "version": candidate_version,
                "comparison": comparison,
            }

    def _promote(self, model_name: str, version: str, metrics: dict) -> None:
        """Promote a model to production status."""
        registry = self._load_registry()

        # Archive current production
        current = registry.get("models", {}).get(model_name)
        if current and current.get("status") == "production":
            current["status"] = "archived"
            current["archived_at"] = datetime.utcnow().isoformat()
            registry["history"].append(current.copy())

        # Set new production
        registry.setdefault("models", {})[model_name] = {
            "model_name": model_name,
            "version": version,
            "status": "production",
            "metrics": metrics,
            "artifact_path": str(self.models_dir / model_name.replace("_xgboost", "_model")),
            "promoted_at": datetime.utcnow().isoformat(),
        }

        self._save_registry(registry)
        logger.info(f"Model PROMOTED to production: {model_name} {version}")

    def get_registry_status(self) -> dict:
        """Get current status of all models in the registry."""
        registry = self._load_registry()
        return {
            "models": registry.get("models", {}),
            "history_count": len(registry.get("history", [])),
        }


class RetrainOrchestrator:
    """Orchestrates the full retrain → evaluate → gate pipeline.

    Connects the feedback store's trigger to the actual retraining
    and model gate evaluation.
    """

    def __init__(self):
        self.gate = ModelGate()

    def retrain_rul(self, feedback_data: list | None = None) -> dict:
        """Retrain the RUL model and evaluate through the gate.

        Args:
            feedback_data: Optional additional training data from feedback.

        Returns:
            Dict with retrain results and promotion decision.
        """
        from machineguard.models.rul_model import RULModel

        logger.info("Starting RUL model retrain...")

        model = RULModel()
        try:
            metrics = model.train()
        except Exception as e:
            logger.error(f"RUL retrain failed: {e}")
            return {"success": False, "error": str(e)}

        # Generate version from timestamp
        version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Register and evaluate
        self.gate.register_candidate("rul_xgboost", version, metrics)
        decision = self.gate.evaluate_and_promote(
            "rul_xgboost", metrics, version
        )

        return {
            "success": True,
            "model": "rul_xgboost",
            "version": version,
            "metrics": metrics,
            "gate_decision": decision,
        }

    def retrain_fault(self, feedback_data: list | None = None) -> dict:
        """Retrain the fault classifier and evaluate through the gate.

        Args:
            feedback_data: Optional additional training data from feedback.

        Returns:
            Dict with retrain results and promotion decision.
        """
        from machineguard.models.fault_classifier import FaultClassifier

        logger.info("Starting fault classifier retrain...")

        model = FaultClassifier()
        try:
            metrics = model.train(use_smote=True)
        except Exception as e:
            logger.error(f"Fault retrain failed: {e}")
            return {"success": False, "error": str(e)}

        version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        self.gate.register_candidate("fault_xgboost", version, metrics)
        decision = self.gate.evaluate_and_promote(
            "fault_xgboost", metrics, version
        )

        return {
            "success": True,
            "model": "fault_xgboost",
            "version": version,
            "metrics": metrics,
            "gate_decision": decision,
        }


# Singletons
model_gate = ModelGate()
retrain_orchestrator = RetrainOrchestrator()
