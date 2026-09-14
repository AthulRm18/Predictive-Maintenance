"""FastAPI application factory for MachineGuard.

Creates the FastAPI app with model loading on startup, CORS middleware,
and route registration.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from machineguard.config import settings, MODELS_DIR
from machineguard.api.schemas import HealthResponse

logger = logging.getLogger(__name__)

# Global model references (loaded on startup)
rul_model = None
fault_model = None
rul_explainer = None
fault_explainer = None


def _load_models() -> None:
    """Load trained models from disk on startup."""
    global rul_model, fault_model, rul_explainer, fault_explainer

    from machineguard.models.rul_model import RULModel
    from machineguard.models.fault_classifier import FaultClassifier
    from machineguard.models.explainer import ModelExplainer

    # Load RUL model
    rul_model_path = MODELS_DIR / "rul_model"
    if (rul_model_path / "model.joblib").exists():
        rul_model = RULModel()
        rul_model.load(rul_model_path)
        logger.info("RUL model loaded successfully.")

        # Create SHAP explainer
        try:
            rul_explainer = ModelExplainer(
                model=rul_model.model,
                feature_names=rul_model.feature_names,
                model_type="regressor",
            )
            logger.info("RUL SHAP explainer created.")
        except Exception as e:
            logger.warning(f"Failed to create RUL explainer: {e}")
    else:
        logger.warning(f"RUL model not found at {rul_model_path}. Train first.")

    # Load Fault classifier
    fault_model_path = MODELS_DIR / "fault_classifier"
    if (fault_model_path / "model.joblib").exists():
        fault_model = FaultClassifier()
        fault_model.load(fault_model_path)
        logger.info("Fault classifier loaded successfully.")

        # Create SHAP explainer
        try:
            fault_explainer = ModelExplainer(
                model=fault_model.model,
                feature_names=fault_model.feature_names,
                model_type="classifier",
                class_names=fault_model.class_names,
            )
            logger.info("Fault SHAP explainer created.")
        except Exception as e:
            logger.warning(f"Failed to create fault explainer: {e}")
    else:
        logger.warning(f"Fault classifier not found at {fault_model_path}. Train first.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler — load models on startup."""
    logger.info("MachineGuard API starting up...")
    _load_models()
    logger.info("Startup complete.")
    yield
    logger.info("MachineGuard API shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade predictive maintenance API. "
            "Provides RUL prediction, fault classification, and SHAP explainability "
            "for multi-fleet industrial equipment."
        ),
        lifespan=lifespan,
    )

    # CORS middleware (for Stage 5 dashboard)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            models_loaded={
                "rul_regressor": rul_model is not None,
                "fault_classifier": fault_model is not None,
            },
        )

    # Register route modules
    from machineguard.api.routes.predict import router as predict_router
    from machineguard.api.routes.explain import router as explain_router
    from machineguard.api.routes.streaming import router as streaming_router

    app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])
    app.include_router(explain_router, prefix="/api/v1", tags=["Explainability"])
    app.include_router(streaming_router, tags=["Streaming"])

    return app


# Module-level app instance (for uvicorn)
app = create_app()
