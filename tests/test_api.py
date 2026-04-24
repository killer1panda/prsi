"""
Unit and integration tests for FastAPI endpoints.
Uses TestClient for synchronous testing and async tests for websockets.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

# Import your FastAPI app - adjust path as needed
try:
    from src.api.api_v2 import app
except ImportError:
    app = None


@pytest.fixture
def client():
    """Fixture providing a TestClient instance."""
    if app is None:
        pytest.skip("FastAPI app not available")
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Mock predictor for isolated endpoint testing."""
    predictor = Mock()
    predictor.predict.return_value = {
        "doom_score": 75.5,
        "risk_level": "high",
        "confidence": 0.92,
        "features": {"toxicity": 0.8, "sentiment": -0.6}
    }
    return predictor


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
    """Test /analyze endpoints."""

    def test_analyze_text_only(self, client):
        payload = {
            "text": "This is absolutely unacceptable behavior from a public figure!",
            "source": "twitter",
            "metadata": {"timestamp": "2026-01-01T00:00:00Z"}
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "doom_score" in data
        assert isinstance(data["doom_score"], (int, float))
        assert 0 <= data["doom_score"] <= 100
        assert "risk_level" in data
        assert data["risk_level"] in ["low", "medium", "high", "critical"]

    def test_analyze_empty_text(self, client):
        payload = {"text": ""}
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422  # Validation error

    def test_analyze_missing_text(self, client):
        payload = {"source": "reddit"}
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_analyze_with_user_id(self, client):
        payload = {
            "text": "Controversial opinion here.",
            "user_id": "test_user_456",
            "include_graph_features": True
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "user_graph_features" in data or "graph_contribution" in data

    def test_analyze_with_image(self, client):
        """Test analyze with image upload (multipart/form-data)."""
        # This requires actual file handling; simplified here
        pass


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

    def test_batch_predict_too_large(self, client):
        payload = {"items": [{"text": f"Post {i}"} for i in range(1001)]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422


class TestAttackEndpoints:
    """Test adversarial attack simulator endpoints."""

    def test_attack_simulate(self, client):
        payload = {
            "text": "I disagree with this policy decision.",
            "strategy": "semantic",
            "num_variants": 3,
            "constraints": {
                "max_toxicity": 0.7,
                "preserve_sentiment": True
            }
        }
        response = client.post("/attack/simulate", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "variants" in data
        assert len(data["variants"]) <= 3
        assert all("text" in v and "doom_score" in v for v in data["variants"])

    def test_attack_simulate_invalid_strategy(self, client):
        payload = {
            "text": "Test",
            "strategy": "invalid_strategy"
        }
        response = client.post("/attack/simulate", json=payload)
        assert response.status_code == 422


class TestDashboardEndpoints:
    """Test dashboard data endpoints."""

    def test_leaderboard(self, client):
        response = client.get("/dashboard/leaderboard?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "leaderboard" in data
        assert len(data["leaderboard"]) <= 10

    def test_privacy_tradeoff(self, client):
        response = client.get("/dashboard/privacy-tradeoff")
        assert response.status_code == 200
        data = response.json()
        assert "epsilon_values" in data
        assert "accuracy_values" in data

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


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_404(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_malformed_json(self, client):
        response = client.post("/analyze", data="not json")
        assert response.status_code == 422

    def test_timeout_simulation(self, client):
        """Test that long requests are handled gracefully."""
        # This would need actual timeout configuration
        pass

    def test_rate_limiting(self, client):
        """Test rate limiting if implemented."""
        # Make many rapid requests
        for _ in range(10):
            response = client.get("/health")
        # After limit, should get 429
        # assert response.status_code == 429
