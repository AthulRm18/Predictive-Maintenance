"""Prediction endpoints for RUL and fault classification."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from machineguard.api.schemas import (
    SensorInput,
    RULPredictionResponse,
    FaultPredictionResponse,
)

router = APIRouter()


@router.post(
    "/predict/rul",
    response_model=RULPredictionResponse,
    summary="Predict Remaining Useful Life",
    description=(
        "Predict the remaining useful life (in cycles) for a turbofan engine "
        "based on its current sensor readings. Trained on NASA C-MAPSS FD001."
    ),
)
async def predict_rul(input_data: SensorInput) -> RULPredictionResponse:
    """Predict RUL for a machine based on sensor readings."""
    from machineguard.api.app import rul_model

    if rul_model is None:
        raise HTTPException(
            status_code=503,
            detail="RUL model not loaded. Train the model first.",
        )

    try:
        result = rul_model.predict(input_data.sensor_readings)

        return RULPredictionResponse(
            machine_id=input_data.machine_id,
            predicted_rul=result["predicted_rul"],
            rul_cap=result["rul_cap"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/predict/fault",
    response_model=FaultPredictionResponse,
    summary="Classify Fault Type",
    description=(
        "Classify the failure mode for a manufacturing machine based on its "
        "sensor readings. Returns probabilities for each failure mode and an "
        "anomaly score. Trained on AI4I 2020 dataset."
    ),
)
async def predict_fault(input_data: SensorInput) -> FaultPredictionResponse:
    """Predict fault type for a machine based on sensor readings."""
    from machineguard.api.app import fault_model

    if fault_model is None:
        raise HTTPException(
            status_code=503,
            detail="Fault classifier not loaded. Train the model first.",
        )

    try:
        result = fault_model.predict(input_data.sensor_readings)

        return FaultPredictionResponse(
            machine_id=input_data.machine_id,
            predicted_class=result["predicted_class"],
            probabilities=result["probabilities"],
            anomaly_score=result["anomaly_score"],
            confidence=result["confidence"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
