"""
Unit and integration tests for FastAPI endpoints.
Uses TestClient for synchronous testing.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

# Import production FastAPI app
try:
    from src.api.api_v2_production import app, config
    config.require_auth = False
except ImportError:
    app = None


@pytest.fixture
def client():
    """Fixture providing a TestClient instance."""
    if app is None:
        pytest.skip("FastAPI app not available")
    return TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "timestamp" in data

    def test_readiness_probe(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_liveness_probe(self, client):
        response = client.get("/live")
        assert response.status_code == 200


class TestAnalyzeEndpoints:
    """Test /analyze and /predict endpoints."""

    def test_predict_text_only(self, client):
        payload = {
            "text": "This is absolutely unacceptable behavior from a public figure!",
            "source": "twitter",
            "metadata": {"timestamp": "2026-01-01T00:00:00Z"}
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "doom_score" in data
        assert isinstance(data["doom_score"], (int, float))
        assert 0 <= data["doom_score"] <= 100
        assert "risk_level" in data
        assert data["risk_level"] in ["low", "medium", "high", "critical"]

    def test_predict_empty_text(self, client):
        payload = {"text": ""}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_missing_text(self, client):
        payload = {"source": "reddit"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestBatchEndpoints:
    """Test batch prediction endpoints."""

    def test_batch_predict(self, client):
        payload = {
            "items": [
                {"text": "Post 1", "user_id": "u1"},
                {"text": "Post 2", "user_id": "u2"},
                {"text": "Post 3", "user_id": "u3"}
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 3
        assert all("doom_score" in p for p in data["predictions"])

    def test_batch_predict_empty(self, client):
        payload = {"items": []}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422


class TestAttackEndpoints:
    """Test adversarial attack simulator endpoints."""

    def test_attack_simulate(self, client):
        payload = {
            "text": "I disagree with this policy decision and they must step down.",
            "strategy": "semantic",
            "num_variants": 3,
            "toxicity_budget": 0.7,
            "use_genetic": True,
            "min_semantic_similarity": 0.5
        }
        response = client.post("/attack/simulate", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "variants" in data
        assert "original_doom_score" in data
        assert len(data["variants"]) <= 3
        assert all("text" in v and "doom_score" in v for v in data["variants"])


class TestDashboardEndpoints:
    """Test dashboard data endpoints."""

    def test_leaderboard(self, client):
        response = client.get("/dashboard/leaderboard?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "leaderboard" in data
        assert len(data["leaderboard"]) <= 10

    def test_drift_status(self, client):
        response = client.get("/dashboard/drift-status")
        assert response.status_code == 200
        data = response.json()
        assert "drift_detected" in data
        assert "overall_risk" in data


class TestPrivacyEndpoints:
    """Test privacy-related endpoints."""

    def test_dp_status(self, client):
        response = client.get("/privacy/dp-status")
        assert response.status_code == 200
        data = response.json()
        assert "epsilon" in data
        assert "delta" in data

    def test_fl_status(self, client):
        response = client.get("/privacy/fl-status")
        assert response.status_code == 200
        data = response.json()
        assert "num_clients" in data
        assert "current_round" in data
