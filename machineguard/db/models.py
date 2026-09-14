"""SQLAlchemy ORM models for TimescaleDB.

Defines the schema for sensor readings, predictions, and the HITL review queue.
Uses JSONB for sensor_data to support schema-agnostic storage across fleets.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class SensorReading(Base):
    """Raw sensor readings from any fleet.

    Uses JSONB for sensor_data so C-MAPSS's 21 sensors and AI4I's 6 features
    both fit without schema changes. This is the production-grade design point:
    new machine types can be added without DDL changes.

    TimescaleDB hypertable on timestamp for efficient time-range queries.
    """

    __tablename__ = "sensor_readings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    machine_id = Column(String(128), nullable=False, index=True)
    fleet_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    sensor_data = Column(JSONB, nullable=False)
    operating_conditions = Column(JSONB)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    predictions = relationship("Prediction", back_populates="reading")

    __table_args__ = (
        Index("ix_sensor_readings_machine_time", "machine_id", "timestamp"),
        Index("ix_sensor_readings_fleet", "fleet_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<SensorReading(id={self.id}, machine_id={self.machine_id}, "
            f"fleet_id={self.fleet_id}, timestamp={self.timestamp})>"
        )


class Prediction(Base):
    """Model predictions with full lineage tracking.

    Stores the model version, prediction output, SHAP values, and confidence.
    Linked to the source sensor reading for traceability.
    """

    __tablename__ = "predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    reading_id = Column(BigInteger, ForeignKey("sensor_readings.id"), nullable=True)
    machine_id = Column(String(128), nullable=False, index=True)
    model_name = Column(String(64), nullable=False)  # "rul_regressor" | "fault_classifier"
    model_version = Column(String(64), nullable=False)
    prediction = Column(JSONB, nullable=False)  # {"rul": 42} or {"class": "HDF", ...}
    confidence = Column(Float)
    anomaly_score = Column(Float)
    shap_values = Column(JSONB)  # Top-k SHAP values
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    reading = relationship("SensorReading", back_populates="predictions")
    review_items = relationship("ReviewItem", back_populates="prediction")

    __table_args__ = (
        Index("ix_predictions_machine", "machine_id"),
        Index("ix_predictions_model", "model_name", "model_version"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, machine_id={self.machine_id}, "
            f"model={self.model_name}:{self.model_version})>"
        )


class ReviewItem(Base):
    """Human-in-the-loop review queue entry.

    Predictions with uncertain confidence or model disagreement are routed
    here instead of auto-alerting. Technicians label events, and their
    verdicts feed back into retraining.

    Schema defined in Stage 1, populated in Stage 3.
    """

    __tablename__ = "review_queue"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    prediction_id = Column(BigInteger, ForeignKey("predictions.id"), nullable=False)
    status = Column(
        String(32),
        nullable=False,
        default="pending",
    )  # pending | confirmed_fault | false_alarm | monitoring
    reviewer_id = Column(String(128))
    fault_type_tag = Column(String(64))
    notes = Column(Text)
    model_version_at_prediction = Column(String(64))
    reviewed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    prediction = relationship("Prediction", back_populates="review_items")

    __table_args__ = (
        Index("ix_review_queue_status", "status"),
        Index("ix_review_queue_prediction", "prediction_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<ReviewItem(id={self.id}, prediction_id={self.prediction_id}, "
            f"status={self.status})>"
        )


class ModelRegistry(Base):
    """Local model registry for tracking trained models.

    Supplements MLflow with lightweight local tracking. Stores model
    performance metrics and promotion status.
    """

    __tablename__ = "model_registry"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(64), nullable=False)
    status = Column(String(32), default="candidate")  # candidate | production | archived
    metrics = Column(JSONB)
    artifact_path = Column(String(512))
    promoted_at = Column(DateTime)
    promoted_by = Column(String(128))
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_model_registry_name_version", "model_name", "model_version", unique=True),
        Index("ix_model_registry_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry(model={self.model_name}:{self.model_version}, "
            f"status={self.status})>"
        )
