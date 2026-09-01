"""
Integration tests for the complete Doom Index pipeline.
Tests end-to-end flows: data ingestion -> feature extraction -> prediction -> API response.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


class TestDataPipeline:
    """Test data ingestion and preprocessing pipeline."""

    def test_preprocessor_pipeline(self):
        """Test DataPreprocessor pipeline."""
        from src.data.preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor()
        raw_posts = [
            {
                "post_id": "p1",
                "text": "Check this out http://example.com #viral @user1",
                "author": "u1",
            },
            {"post_id": "p2", "text": "Normal discussion post without links", "author": "u2"},
        ]

        cleaned_posts = preprocessor.preprocess_pipeline(raw_posts)
        assert len(cleaned_posts) == 2
        assert "http" not in cleaned_posts[0]["cleaned_text"]
        assert cleaned_posts[0]["post_id"] == "p1"

    def test_feature_engineering_pipeline(self):
        """Test sentiment and toxicity feature extraction."""
        from src.features.sentiment import analyze_text_sentiment
        from src.features.toxicity import analyze_text_toxicity

        text_neg = "I absolutely hate this and everyone should cancel them immediately!!!"
        text_pos = "Lovely weather today, hope everyone is having a wonderful day."

        sent_neg = analyze_text_sentiment(text_neg)
        sent_pos = analyze_text_sentiment(text_pos)
        tox_neg = analyze_text_toxicity(text_neg)

        assert sent_neg["sentiment_compound"] < sent_pos["sentiment_compound"]
        score = tox_neg.get("toxicity_score", tox_neg.get("toxicity", 0.0))
        assert score > 0.2

    def test_database_connectors(self):
        """Test MongoDB and Neo4j connectors with self-healing fallback."""
        from src.data.db_connectors import get_mongodb, get_neo4j

        mongo = get_mongodb()
        assert mongo is not None
        post_id = mongo.insert_post(
            {"post_id": "test_integration_1", "text": "Integration test post"}
        )
        assert post_id is not None

        neo = get_neo4j()
        assert neo is not None


class TestModelPipeline:
    """Test model training and inference pipeline."""

    def test_distilbert_forward(self):
        """Test DistilBERT model forward pass."""
        from transformers import (DistilBertForSequenceClassification,
                                  DistilBertTokenizer)

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        inputs = tokenizer("Test post about cancellation", return_tensors="pt", padding=True)
        outputs = model(**inputs)

        assert outputs.logits.shape == (1, 2)
        assert not torch.isnan(outputs.logits).any()

    def test_gnn_forward(self):
        """Test GraphSAGE encoder forward pass."""
        from src.models.gnn_model import GraphSAGEEncoder

        model = GraphSAGEEncoder(in_channels=16, hidden_channels=32, out_channels=32, num_layers=2)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)

        out = model(x, edge_index)
        assert out.shape == (10, 32)

    def test_multimodal_fusion(self):
        """Test CrossModalAttention fusion of text and graph embeddings."""
        from src.models.fusion import CrossModalAttention

        text_emb = torch.randn(4, 768)
        graph_emb = torch.randn(4, 128)

        fusion = CrossModalAttention(graph_dim=128, text_dim=768, num_heads=8)
        fused_graph, fused_text = fusion(graph_emb, text_emb)
        assert fused_graph.shape[0] == 4
        assert fused_text.shape[0] == 4


class TestPrivacyPipeline:
    """Test privacy-preserving components."""

    def test_differential_privacy_noise(self):
        """Test that DP Gaussian noise is applied."""
        from src.privacy.dp_trainer import add_gaussian_noise

        tensor = torch.ones(100)
        noisy = add_gaussian_noise(tensor, sigma=0.5, clip_norm=1.0)

        assert not torch.equal(tensor, noisy)
        assert torch.std(noisy) > 0.01

    def test_federated_aggregation(self):
        """Test FedAvg layer averaging."""
        from src.privacy.fl_simulator import federated_averaging

        w1 = [np.ones((4, 4), dtype=np.float32), np.zeros((4,), dtype=np.float32)]
        w2 = [np.ones((4, 4), dtype=np.float32) * 3, np.ones((4,), dtype=np.float32) * 2]

        avg = federated_averaging([w1, w2], sample_counts=[10, 10])
        assert np.allclose(avg[0], np.ones((4, 4)) * 2.0)
        assert np.allclose(avg[1], np.ones((4,)) * 1.0)


class TestStreamingPipeline:
    """Test streaming and feature store components."""

    def test_kafka_pipeline_init(self):
        """Test Kafka pipeline initialization."""
        from src.streaming.kafka_pipeline import KafkaConfig, KafkaPipeline

        def mock_predictor(posts):
            return [{"doom_score": 50.0, "risk_level": "medium"}]

        # Verified config initializes without syntax or timeout errors
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        assert config.max_poll_interval_ms >= config.session_timeout_ms

    def test_feature_store_consistency(self):
        """Test online/offline feature store consistency."""
        from src.features.feature_store import FeatureStore, FeatureView

        store = FeatureStore(redis_host="localhost", offline_path="/tmp/test_fs")

        view = FeatureView(
            name="user_features",
            entities=["user_id"],
            features=["follower_count", "avg_sentiment"],
            online=True,
        )
        store.register_feature_view(view)

        # Push online
        store.push_online(
            "user", "u123", "user_features", {"follower_count": 1000, "avg_sentiment": -0.2}
        )

        features = store.get_online("user", "u123", "user_features")
        assert features["follower_count"] == 1000
        assert features["avg_sentiment"] == -0.2
