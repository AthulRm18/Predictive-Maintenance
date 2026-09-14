"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from machineguard.api.app import app
    return TestClient(app)


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data

    def test_health_shows_model_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "rul_regressor" in data["models_loaded"]
        assert "fault_classifier" in data["models_loaded"]


class TestPredictEndpoints:
    """Test prediction endpoints (may return 503 if models not loaded)."""

    def test_predict_rul_without_model(self, client):
        """Should return 503 if model not trained."""
        response = client.post(
            "/api/v1/predict/rul",
            json={
                "machine_id": "test_unit_1",
                "sensor_readings": {"sensor_2": 0.5, "sensor_3": 0.6},
            },
        )
        # Either 200 (model loaded) or 503 (model not loaded)
        assert response.status_code in [200, 503]

    def test_predict_fault_without_model(self, client):
        """Should return 503 if model not trained."""
        response = client.post(
            "/api/v1/predict/fault",
            json={
                "machine_id": "test_machine_1",
                "sensor_readings": {
                    "Air temperature [K]": 300.0,
                    "Process temperature [K]": 310.0,
                    "Rotational speed [rpm]": 1500.0,
                    "Torque [Nm]": 40.0,
                    "Tool wear [min]": 100.0,
                },
            },
        )
        assert response.status_code in [200, 503]

    def test_predict_rul_invalid_body(self, client):
        """Should return 422 for invalid request body."""
        response = client.post(
            "/api/v1/predict/rul",
            json={"invalid": "data"},
        )
        assert response.status_code == 422


class TestExplainEndpoints:
    """Test explanation endpoints."""

    def test_explain_rul_without_model(self, client):
        response = client.post(
            "/api/v1/explain/rul",
            json={
                "machine_id": "test_unit_1",
                "sensor_readings": {"sensor_2": 0.5},
            },
        )
        assert response.status_code in [200, 503]

    def test_explain_fault_without_model(self, client):
        response = client.post(
            "/api/v1/explain/fault",
            json={
                "machine_id": "test_machine_1",
                "sensor_readings": {"Air temperature [K]": 300.0},
            },
        )
        assert response.status_code in [200, 503]
