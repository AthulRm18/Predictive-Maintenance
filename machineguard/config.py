"""Centralized configuration for MachineGuard."""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field


# Project root is the parent of the machineguard package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"


class DatabaseSettings(BaseSettings):
    """TimescaleDB connection settings."""

    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="machineguard", alias="DB_NAME")
    db_user: str = Field(default="machineguard", alias="DB_USER")
    db_password: str = Field(default="machineguard_dev", alias="DB_PASSWORD")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    model_config = {"env_prefix": "", "extra": "ignore"}


class MLflowSettings(BaseSettings):
    """MLflow tracking settings."""

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="machineguard", alias="MLFLOW_EXPERIMENT_NAME"
    )

    model_config = {"env_prefix": "", "extra": "ignore"}


class ModelSettings(BaseSettings):
    """Model training & inference settings."""

    # RUL model
    rul_cap: int = Field(default=125, description="Max RUL value (piece-wise linear cap)")
    rul_model_name: str = "rul_xgboost"

    # Fault classifier
    fault_model_name: str = "fault_xgboost"
    fault_classes: list[str] = [
        "no_failure", "TWF", "HDF", "PWF", "OSF", "RNF"
    ]

    # SHAP
    shap_top_k: int = Field(default=10, description="Number of top SHAP features to return")

    # Uncertainty thresholds (for Stage 3 HITL)
    uncertainty_low: float = 0.3
    uncertainty_high: float = 0.7

    model_config = {"extra": "ignore"}


class AppSettings(BaseSettings):
    """Top-level application settings."""

    app_name: str = "MachineGuard"
    app_version: str = "0.1.0"
    debug: bool = Field(default=True, alias="DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    db: DatabaseSettings = DatabaseSettings()
    mlflow: MLflowSettings = MLflowSettings()
    model: ModelSettings = ModelSettings()

    model_config = {"env_prefix": "", "extra": "ignore"}


# Singleton settings instance
settings = AppSettings()
