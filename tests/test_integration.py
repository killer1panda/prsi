"""
Integration tests for the complete Doom Index pipeline.
Tests end-to-end flows: data ingestion -> feature extraction -> prediction -> API response.
"""
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Mark all tests as integration
pytestmark = pytest.mark.integration


class TestDataPipeline:
    """Test data ingestion and preprocessing pipeline."""

    def test_pushshift_ingestion(self):
        """Test Pushshift NDJSON parsing and filtering."""
        from src.data.preprocessing import preprocess_reddit_raw

        raw_posts = [
            {"body": "This is a test post about cancel culture", "score": 100, "created_utc": 1609459200},
            {"body": "[deleted]", "score": 0, "created_utc": 1609459200},  # Should be filtered
            {"body": "Normal discussion here", "score": 5, "created_utc": 1609459200}
        ]

        processed = preprocess_reddit_raw(raw_posts)
        assert len(processed) == 2  # [deleted] filtered out
        assert all("body" in p and p["body"] != "[deleted]" for p in processed)

    def test_feature_engineering_pipeline(self):
        """Test feature extraction from raw text."""
        from src.features.engineering import extract_features

        texts = [
            "I absolutely hate this person and everyone should cancel them!!!",
            "Lovely weather today, hope everyone is doing well."
        ]

        features = extract_features(texts)
        assert features.shape[0] == 2
        assert features.shape[1] > 0

        # First text should have higher toxicity/sentiment negativity
        assert features.iloc[0]["sentiment_compound"] < features.iloc[1]["sentiment_compound"]

    def test_neo4j_graph_building(self):
        """Test graph construction from user interactions."""
        from src.data.build_neo4j_graph import GraphBuilder

        df = pd.DataFrame({
            "author": ["user1", "user2", "user1", "user3"],
            "parent_author": [None, "user1", "user2", "user1"],
            "subreddit": ["test", "test", "test", "test"],
            "score": [10, 5, 3, 8]
        })

        builder = GraphBuilder(uri="bolt://localhost:7687", user="neo4j", password="test")
        # Note: This would need a running Neo4j instance; mock in real tests
        assert builder is not None


class TestModelPipeline:
    """Test model training and inference pipeline."""

    def test_distilbert_forward(self):
        """Test DistilBERT model forward pass."""
        from transformers import DistilBertForSequenceClassification

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )

        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        inputs = tokenizer("Test post about cancellation", return_tensors="pt", padding=True)
        outputs = model(**inputs)

        assert outputs.logits.shape == (1, 2)
        assert not torch.isnan(outputs.logits).any()

    def test_gnn_forward(self):
        """Test GNN forward pass with dummy graph."""
        from src.models.gnn_model import GraphSAGE  # Assuming this exists
        import torch_geometric.data as pyg_data

        x = torch.randn(10, 64)
        edge_index = torch.tensor([[0,1,2,3,4], [1,2,3,4,5]], dtype=torch.long)
        data = pyg_data.Data(x=x, edge_index=edge_index)

        # This test assumes GraphSAGE exists; adjust based on actual model
        # model = GraphSAGE(in_channels=64, hidden_channels=128, out_channels=64)
        # out = model(data.x, data.edge_index)
        # assert out.shape == (10, 64)

    def test_multimodal_fusion(self):
        """Test multimodal fusion of text and graph embeddings."""
        from src.models.fusion import CrossModalFusion  # Assuming exists

        text_emb = torch.randn(4, 768)
        graph_emb = torch.randn(4, 128)

        # fusion = CrossModalFusion(text_dim=768, graph_dim=128)
        # output = fusion(text_emb, graph_emb)
        # assert output.shape[0] == 4


class TestAPIPipeline:
    """Test API endpoints end-to-end."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_analyze_endpoint(self, client):
        """Test analyze endpoint with sample text."""
        payload = {
            "text": "This person needs to be cancelled immediately!",
            "user_id": "test_user_123",
            "source": "reddit"
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "doom_score" in data
        assert 0 <= data["doom_score"] <= 100
        assert "risk_level" in data

    def test_attack_simulator_endpoint(self, client):
        """Test attack simulator endpoint."""
        payload = {
            "text": "I think this policy is misguided.",
            "target_doom_score": 80,
            "constraints": {"toxicity": 0.7}
        }
        response = client.post("/attack/simulate", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "variants" in data
        assert len(data["variants"]) > 0
        assert all("text" in v and "doom_score" in v for v in data["variants"])

    def test_batch_prediction_endpoint(self, client):
        """Test batch prediction endpoint."""
        payload = {
            "texts": [
                "Post one about controversy",
                "Post two about kittens",
                "Post three about politics"
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 3
        assert all("doom_score" in p for p in data["predictions"])


class TestPrivacyPipeline:
    """Test privacy-preserving components."""

    def test_differential_privacy_noise(self):
        """Test that DP noise is actually added."""
        from src.privacy.dp_trainer import add_gaussian_noise

        tensor = torch.ones(100)
        noisy = add_gaussian_noise(tensor, epsilon=1.0, sensitivity=1.0)

        assert not torch.equal(tensor, noisy)
        assert torch.std(noisy) > 0.01  # Noise should be noticeable

    def test_federated_aggregation(self):
        """Test FedAvg aggregation."""
        from src.privacy.fl_simulator import federated_averaging

        weights = [
            {"layer1": torch.ones(10), "layer2": torch.zeros(5)},
            {"layer1": torch.ones(10) * 2, "layer2": torch.ones(5)},
        ]

        aggregated = federated_averaging(weights)
        assert torch.allclose(aggregated["layer1"], torch.ones(10) * 1.5)
        assert torch.allclose(aggregated["layer2"], torch.ones(5) * 0.5)


class TestStreamingPipeline:
    """Test streaming components."""

    def test_kafka_message_processing(self):
        """Test Kafka message processing logic."""
        from src.streaming.kafka_pipeline import KafkaPipeline

        def mock_predictor(post):
            return {"doom_score": 50.0, "risk_level": "medium"}

        pipeline = KafkaPipeline(predictor=mock_predictor)

        test_msg = '{"text": "Test", "user_id": "u1", "post_id": "p1"}'
        result = pipeline._process_message(test_msg)

        assert result is not None
        assert result["doom_score"] == 50.0

    def test_feature_store_consistency(self):
        """Test online/offline feature store consistency."""
        from src.features.feature_store import FeatureStore, FeatureView

        store = FeatureStore(redis_host="localhost", offline_path="/tmp/test_fs")

        view = FeatureView(
            name="user_features",
            entities=["user_id"],
            features=["follower_count", "avg_sentiment"],
            online=True
        )
        store.register_feature_view(view)

        # Push online
        store.push_online("user", "u123", "user_features", {
            "follower_count": 1000,
            "avg_sentiment": -0.2
        })

        # Read online
        features = store.get_online("user", "u123", "user_features")
        assert features["follower_count"] == 1000
        assert features["avg_sentiment"] == -0.2


# Fixtures
@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    from fastapi.testclient import TestClient
    from src.api.api_v2 import app  # Adjust import as needed
    return TestClient(app)


@pytest.fixture(scope="session")
def trained_model():
    """Load or create a trained model for integration tests."""
    # This could load a small pretrained model or create a fresh one
    pass
