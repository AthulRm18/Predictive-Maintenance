"""Explainability endpoints — SHAP explanations for predictions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from machineguard.api.schemas import (
    SensorInput,
    RULExplanationResponse,
    FaultExplanationResponse,
    SHAPFeature,
)

router = APIRouter()


@router.post(
    "/explain/rul",
    response_model=RULExplanationResponse,
    summary="Explain RUL Prediction",
    description=(
        "Get SHAP-based explanation for a RUL prediction. Returns per-feature "
        "contributions showing which sensors are driving the RUL estimate up or down."
    ),
)
async def explain_rul(input_data: SensorInput) -> RULExplanationResponse:
    """Generate SHAP explanation for a RUL prediction."""
    from machineguard.api.app import rul_explainer

    if rul_explainer is None:
        raise HTTPException(
            status_code=503,
            detail="RUL explainer not available. Train the model first.",
        )

    try:
        explanation = rul_explainer.explain_single(input_data.sensor_readings)

        top_features = [
            SHAPFeature(**feat) for feat in explanation["top_features"]
        ]

        return RULExplanationResponse(
            machine_id=input_data.machine_id,
            shap_values=explanation["shap_values"],
            top_features=top_features,
            base_value=explanation["base_value"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Explanation failed: {str(e)}"
        )


@router.post(
    "/explain/fault",
    response_model=FaultExplanationResponse,
    summary="Explain Fault Classification",
    description=(
        "Get SHAP-based explanation for a fault classification. Returns per-feature "
        "contributions for the predicted failure mode, plus top features per class."
    ),
)
async def explain_fault(input_data: SensorInput) -> FaultExplanationResponse:
    """Generate SHAP explanation for a fault classification."""
    from machineguard.api.app import fault_explainer

    if fault_explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Fault explainer not available. Train the model first.",
        )

    try:
        explanation = fault_explainer.explain_single(input_data.sensor_readings)

        top_features = [
            SHAPFeature(**feat) for feat in explanation["top_features"]
        ]

        return FaultExplanationResponse(
            machine_id=input_data.machine_id,
            predicted_class=explanation.get("predicted_class", "unknown"),
            shap_values=explanation["shap_values"],
            top_features=top_features,
            base_value=explanation["base_value"],
            per_class_top_features=explanation.get("per_class_top_features", {}),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Explanation failed: {str(e)}"
        )
