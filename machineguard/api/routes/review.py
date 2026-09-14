"""Review queue API routes — HITL feedback endpoints.

Provides endpoints for:
- Viewing pending review items
- Submitting human verdicts
- Checking feedback statistics
- Triggering model retraining
- Viewing model registry status
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from machineguard.api.schemas import (
    SubmitReviewRequest,
    SubmitReviewResponse,
    FeedbackStatsResponse,
    RetrainRequest,
    RetrainResponse,
    ModelRegistryResponse,
    ReviewQueueItem,
)
from machineguard.hitl.router import ReviewRouter, RoutingDecision
from machineguard.hitl.feedback_store import feedback_store
from machineguard.hitl.retrain_gate import model_gate, retrain_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory review queue (would be backed by DB in production with TimescaleDB)
_review_queue: dict[str, dict] = {}
_review_router = ReviewRouter()


@router.post("/review/route", tags=["HITL"])
async def route_prediction(prediction: dict[str, Any]):
    """Route a prediction through the HITL router.

    The router examines the prediction and decides whether it needs
    human review based on confidence, anomaly score, and other factors.

    Returns the routing decision and, if review is needed, creates a
    review queue entry.
    """
    pred_type = prediction.get("type", "")
    decision: RoutingDecision | None = None

    if pred_type == "fault":
        decision = _review_router.check_fault_prediction(prediction)
    elif pred_type == "rul":
        decision = _review_router.check_rul_prediction(prediction)

    if decision is None:
        return {
            "needs_review": False,
            "reason": "unknown_prediction_type",
        }

    result = {
        "needs_review": decision.needs_review,
        "reasons": [r.value for r in decision.reasons],
        "priority": decision.priority,
        "auto_alert": decision.auto_alert,
    }

    if decision.needs_review:
        # Create review queue entry
        item_id = str(uuid.uuid4())[:8]
        queue_entry = {
            "id": item_id,
            "prediction_id": prediction.get("prediction_id", item_id),
            "prediction": prediction,
            "routing_reasons": [r.value for r in decision.reasons],
            "priority": decision.priority,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
        }
        _review_queue[item_id] = queue_entry
        result["queue_item_id"] = item_id
        logger.info(f"Prediction routed to review: {item_id} ({decision.reasons})")

    return result


@router.get("/review/queue", response_model=list[ReviewQueueItem], tags=["HITL"])
async def get_review_queue(
    status: str | None = None,
    priority: str | None = None,
    limit: int = 50,
):
    """Get pending items in the review queue.

    Args:
        status: Filter by status (pending, reviewed).
        priority: Filter by priority (low, medium, high, critical).
        limit: Max items to return.
    """
    items = list(_review_queue.values())

    if status:
        items = [i for i in items if i["status"] == status]
    if priority:
        items = [i for i in items if i["priority"] == priority]

    # Sort by priority (critical first)
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    items.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 4))

    return items[:limit]


@router.get("/review/queue/{item_id}", tags=["HITL"])
async def get_review_item(item_id: str):
    """Get a specific review queue item."""
    item = _review_queue.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Review item {item_id} not found")
    return item


@router.post("/review/submit", response_model=SubmitReviewResponse, tags=["HITL"])
async def submit_review(request: SubmitReviewRequest):
    """Submit a human verdict on a prediction.

    Records the feedback and checks if enough corrections have
    accumulated to trigger model retraining.
    """
    # Find the prediction in the queue
    queue_item = None
    for item in _review_queue.values():
        if item.get("prediction_id") == request.prediction_id:
            queue_item = item
            break

    prediction = queue_item.get("prediction", {}) if queue_item else {}
    fleet_id = prediction.get("fleet_id", "unknown")

    # Record feedback
    try:
        feedback_store.add_feedback(
            prediction_id=request.prediction_id,
            prediction=prediction,
            verdict=request.verdict,
            corrected_label=request.corrected_label,
            reviewer_id=request.reviewer_id,
            sensor_readings=prediction.get("sensor_readings"),
            notes=request.notes,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update queue item status
    if queue_item:
        queue_item["status"] = "reviewed"
        queue_item["verdict"] = request.verdict
        queue_item["reviewed_at"] = datetime.utcnow().isoformat()

    # Check if retrain should be triggered
    should_retrain, retrain_stats = feedback_store.should_trigger_retrain(fleet_id)

    return SubmitReviewResponse(
        status="recorded",
        prediction_id=request.prediction_id,
        verdict=request.verdict,
        retrain_check={
            "fleet_id": fleet_id,
            "should_retrain": should_retrain,
            **retrain_stats,
        },
    )


@router.get("/review/stats/{fleet_id}", response_model=FeedbackStatsResponse, tags=["HITL"])
async def get_feedback_stats(fleet_id: str):
    """Get feedback statistics for a fleet."""
    should_retrain, stats = feedback_store.should_trigger_retrain(fleet_id)

    return FeedbackStatsResponse(
        fleet_id=fleet_id,
        total_feedback=feedback_store.get_feedback_count(fleet_id),
        verdicts=stats.get("verdicts", {}),
        should_retrain=should_retrain,
        retrain_stats=stats,
    )


@router.post("/review/retrain", response_model=RetrainResponse, tags=["HITL"])
async def trigger_retrain(request: RetrainRequest):
    """Trigger model retraining with the model promotion gate.

    The retrained model is compared against the current production model.
    It is only promoted if it shows meaningful improvement.
    """
    if request.model_type not in ("rul", "fault"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type '{request.model_type}'. Must be 'rul' or 'fault'."
        )

    if request.model_type == "rul":
        result = retrain_orchestrator.retrain_rul()
    else:
        result = retrain_orchestrator.retrain_fault()

    return RetrainResponse(
        success=result.get("success", False),
        model=result.get("model", request.model_type),
        version=result.get("version"),
        metrics=result.get("metrics", {}),
        gate_decision=result.get("gate_decision", {}),
        error=result.get("error"),
    )


@router.get("/models/registry", response_model=ModelRegistryResponse, tags=["HITL"])
async def get_model_registry():
    """Get current model registry status."""
    status = model_gate.get_registry_status()
    return ModelRegistryResponse(
        models=status.get("models", {}),
        history_count=status.get("history_count", 0),
    )
