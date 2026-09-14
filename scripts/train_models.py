"""Training script for MachineGuard models.

Trains both the RUL regressor (C-MAPSS) and fault classifier (AI4I),
saves artifacts, and logs metrics.

Usage:
    python scripts/train_models.py [--rul-only | --fault-only]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from machineguard.models.rul_model import RULModel
from machineguard.models.fault_classifier import FaultClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def train_rul_model() -> dict:
    """Train the C-MAPSS RUL regressor."""
    logger.info("=" * 60)
    logger.info("Training RUL Model (C-MAPSS FD001)")
    logger.info("=" * 60)

    model = RULModel()
    metrics = model.train(subset="FD001")

    save_path = model.save()
    logger.info(f"Model saved to: {save_path}")

    # Print feature importance
    importance = model.get_feature_importance(top_k=10)
    logger.info("\nTop 10 features by importance:")
    for rank, (name, score) in enumerate(importance, 1):
        logger.info(f"  {rank}. {name}: {score:.4f}")

    return metrics


def train_fault_classifier() -> dict:
    """Train the AI4I fault classifier."""
    logger.info("=" * 60)
    logger.info("Training Fault Classifier (AI4I 2020)")
    logger.info("=" * 60)

    model = FaultClassifier()
    metrics = model.train(use_smote=True)

    save_path = model.save()
    logger.info(f"Model saved to: {save_path}")

    # Print feature importance
    importance = model.get_feature_importance(top_k=10)
    logger.info("\nTop 10 features by importance:")
    for rank, (name, score) in enumerate(importance, 1):
        logger.info(f"  {rank}. {name}: {score:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MachineGuard models")
    parser.add_argument("--rul-only", action="store_true", help="Train only the RUL model")
    parser.add_argument("--fault-only", action="store_true", help="Train only the fault classifier")
    args = parser.parse_args()

    results = {}

    if not args.fault_only:
        try:
            results["rul"] = train_rul_model()
        except FileNotFoundError as e:
            logger.error(f"C-MAPSS data not found: {e}")
            logger.error("Run 'python scripts/download_datasets.py' first.")
        except Exception as e:
            logger.error(f"RUL training failed: {e}", exc_info=True)

    if not args.rul_only:
        try:
            results["fault"] = train_fault_classifier()
        except FileNotFoundError as e:
            logger.error(f"AI4I data not found: {e}")
            logger.error("Run 'python scripts/download_datasets.py' first.")
        except Exception as e:
            logger.error(f"Fault classifier training failed: {e}", exc_info=True)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    if "rul" in results:
        test = results["rul"].get("test", {})
        logger.info(f"\nRUL Model:")
        logger.info(f"  Test RMSE: {test.get('rmse', 'N/A')}")
        logger.info(f"  Test Asymmetric Score: {test.get('asymmetric_score', 'N/A')}")
        logger.info(f"  Early/Late split: {test.get('pct_early', 'N/A')}% / {test.get('pct_late', 'N/A')}%")

    if "fault" in results:
        test = results["fault"].get("test", {})
        logger.info(f"\nFault Classifier:")
        logger.info(f"  Macro Recall: {test.get('macro_recall', 'N/A')}")
        logger.info(f"  Macro F1: {test.get('macro_f1', 'N/A')}")
        logger.info(f"  Per-class recall: {test.get('per_class_recall', 'N/A')}")

    # Save summary
    summary_path = project_root / "models" / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        # Convert numpy types for JSON serialization
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nFull metrics saved to: {summary_path}")


if __name__ == "__main__":
    main()
