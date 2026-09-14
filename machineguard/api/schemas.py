"""Pydantic request/response models for the MachineGuard API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---- Request Models ----

class SensorInput(BaseModel):
    """Sensor reading input for prediction."""
    machine_id: str = Field(..., description="Unique machine identifier")
    sensor_readings: dict[str, float] = Field(
        ..., description="Dict of sensor_name → value"
    )
    operating_conditions: dict[str, Any] = Field(
        default_factory=dict,
        description="Operating regime info (e.g., {'type': 'L'})"
    )


# ---- Response Models ----

class RULPredictionResponse(BaseModel):
    """RUL prediction result."""
    machine_id: str
    predicted_rul: float = Field(..., description="Predicted remaining useful life in cycles")
    rul_cap: int = Field(..., description="Maximum RUL cap used in training")
    model_version: str = "v0.1.0"


class FaultPredictionResponse(BaseModel):
    """Fault classification result."""
    machine_id: str
    predicted_class: str = Field(..., description="Predicted failure mode")
    probabilities: dict[str, float] = Field(
        ..., description="Probability for each class"
    )
    anomaly_score: float = Field(
        ..., description="1 - P(no_failure), higher = more anomalous"
    )
    confidence: float = Field(..., description="Max class probability")
    model_version: str = "v0.1.0"


class SHAPFeature(BaseModel):
    """Single feature's SHAP contribution."""
    feature: str
    shap_value: float
    feature_value: float = 0.0
    direction: str = ""


class RULExplanationResponse(BaseModel):
    """SHAP explanation for RUL prediction."""
    machine_id: str
    shap_values: dict[str, float]
    top_features: list[SHAPFeature]
    base_value: float
    model_type: str = "regressor"


class FaultExplanationResponse(BaseModel):
    """SHAP explanation for fault classification."""
    machine_id: str
    predicted_class: str
    shap_values: dict[str, float]
    top_features: list[SHAPFeature]
    base_value: float
    per_class_top_features: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    model_type: str = "classifier"


class HealthResponse(BaseModel):
    """API health check response."""
    status: str = "healthy"
    version: str
    models_loaded: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str = ""


# ---- HITL Models ----

class ReviewQueueItem(BaseModel):
    """A prediction routed to the review queue."""
    id: str
    prediction_id: str
    prediction: dict[str, Any]
    sensor_readings: dict[str, float] | None = None
    routing_reasons: list[str]
    priority: str
    status: str = "pending"
    created_at: str


class SubmitReviewRequest(BaseModel):
    """Human verdict on a prediction."""
    prediction_id: str = Field(..., description="ID of the prediction to review")
    verdict: str = Field(
        ...,
        description="One of: confirmed_fault, false_alarm, monitoring, corrected"
    )
    corrected_label: str | None = Field(
        None,
        description="If verdict is 'corrected', the true label"
    )
    reviewer_id: str = Field(default="anonymous", description="Reviewer identifier")
    notes: str = Field(default="", description="Free-text notes")


class SubmitReviewResponse(BaseModel):
    """Response after submitting a review."""
    status: str
    prediction_id: str
    verdict: str
    retrain_check: dict[str, Any] = Field(default_factory=dict)


class FeedbackStatsResponse(BaseModel):
    """Statistics about accumulated feedback."""
    fleet_id: str
    total_feedback: int
    verdicts: dict[str, int] = Field(default_factory=dict)
    should_retrain: bool
    retrain_stats: dict[str, Any] = Field(default_factory=dict)


class RetrainRequest(BaseModel):
    """Request to trigger model retraining."""
    model_type: str = Field(
        ..., description="Which model to retrain: 'rul' or 'fault'"
    )
    force: bool = Field(
        default=False,
        description="Force retrain even if feedback threshold not met"
    )


class RetrainResponse(BaseModel):
    """Response from a retrain operation."""
    success: bool
    model: str
    version: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    gate_decision: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ModelRegistryResponse(BaseModel):
    """Current status of the model registry."""
    models: dict[str, Any] = Field(default_factory=dict)
    history_count: int = 0
