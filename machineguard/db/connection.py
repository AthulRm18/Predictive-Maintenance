"""TimescaleDB connection pool and session management.

Provides async-safe database session factory and table creation utilities.
TimescaleDB hypertable creation is handled here for the sensor_readings table.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from machineguard.config import settings
from machineguard.db.models import Base

logger = logging.getLogger(__name__)

# Module-level engine and session factory (initialized lazily)
_engine = None
_SessionFactory = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.db.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=settings.debug,
        )
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions with auto-commit/rollback."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Initialize database: create tables and TimescaleDB hypertables.

    Idempotent: safe to call multiple times.
    """
    engine = get_engine()

    # Create all tables
    Base.metadata.create_all(engine)
    logger.info("Database tables created.")

    # Create TimescaleDB hypertable for sensor_readings
    with engine.connect() as conn:
        try:
            # Enable TimescaleDB extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            conn.commit()
            logger.info("TimescaleDB extension enabled.")

            # Convert sensor_readings to hypertable
            conn.execute(text(
                "SELECT create_hypertable('sensor_readings', 'timestamp', "
                "if_not_exists => TRUE, migrate_data => TRUE)"
            ))
            conn.commit()
            logger.info("sensor_readings converted to TimescaleDB hypertable.")

        except Exception as e:
            logger.warning(
                f"TimescaleDB setup skipped (expected if using plain PostgreSQL): {e}"
            )
            conn.rollback()


def drop_all_tables() -> None:
    """Drop all tables. USE WITH CAUTION — development only."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.warning("All database tables dropped.")
