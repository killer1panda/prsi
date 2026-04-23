"""Unit tests for Doom Index v2 multimodal components.

Run with: pytest tests/test_multimodal.py -v
"""

import pytest
import numpy as np
import pandas as pd

# Skip tests if dependencies not installed
try:
    import torch
    from torch_geometric.data import Data
    from src.models.gnn_model import GraphSAGEEncoder, TextEncoder, FusionMLP, MultimodalDoomPredictor
    from src.features.graph_extractor import GraphExtractor
    from src.attacks.adversarial_generator import AdversarialGenerator, AttackResult
    from src.privacy.dp_trainer import DPDoomTrainer
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Define stubs to prevent NameError in decorator expressions
    Data = None


@pytest.mark.skipif(not DEPS_AVAILABLE, reason=f"Dependencies missing: {IMPORT_ERROR}")
class TestGraphSAGEEncoder:
    """Test GraphSAGE encoder."""

    def test_forward_shape(self):
        encoder = GraphSAGEEncoder(in_channels=6, hidden_channels=64, out_channels=128, num_layers=2)
        x = torch.randn(10, 6)
        edge_index = torch.tensor([[0,1,2,3,4], [1,2,3,4,5]], dtype=torch.long)
        out = encoder(x, edge_index)
        assert out.shape == (10, 128)

    def test_get_embeddings(self):
        encoder = GraphSAGEEncoder(in_channels=6, hidden_channels=32, out_channels=64)
        x = torch.randn(5, 6)
        edge_index = torch.tensor([[0,1,2], [1,2,3]], dtype=torch.long)
        emb = encoder.get_embeddings(x, edge_index)
        assert emb.shape == (5, 64)


@pytest.mark.skipif(not DEPS_AVAILABLE, reason=f"Dependencies missing: {IMPORT_ERROR}")
class TestTextEncoder:
    """Test DistilBERT text encoder."""

    def test_output_shape(self):
        encoder = TextEncoder(model_name="distilbert-base-uncased", freeze_layers=6)
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)
        out = encoder(input_ids, attention_mask)
        assert out.shape == (2, 768)


@pytest.mark.skipif(not DEPS_AVAILABLE, reason=f"Dependencies missing: {IMPORT_ERROR}")
class TestFusionMLP:
    """Test fusion MLP."""

    def test_forward(self):
        fusion = FusionMLP(graph_dim=128, text_dim=768, hidden_dim=256, num_classes=2)
        graph_emb = torch.randn(4, 128)
        text_emb = torch.randn(4, 768)
        out = fusion(graph_emb, text_emb)
        assert out.shape == (4, 2)


@pytest.mark.skipif(not DEPS_AVAILABLE, reason=f"Dependencies missing: {IMPORT_ERROR}")
class TestMultimodalDoomPredictor:
    """Test end-to-end multimodal predictor."""

    def test_predict(self):
        model = MultimodalDoomPredictor(graph_in_channels=6, graph_hidden=32, graph_out=32, 
                                        text_freeze=6, fusion_hidden=64)

        x = torch.randn(5, 6)
        edge_index = torch.tensor([[0,1,2], [1,2,3]], dtype=torch.long)

        pred, prob = model.predict(
            x=x, edge_index=edge_index,
            text="This is a test post.",
            user_idx=0,
            device="cpu",
        )

        assert pred in [0, 1]
        assert 0.0 <= prob <= 1.0

    def test_get_multimodal_embeddings(self):
        model = MultimodalDoomPredictor(graph_in_channels=6, graph_hidden=16, graph_out=16,
                                        text_freeze=6, fusion_hidden=32)
        x = torch.randn(3, 6)
        edge_index = torch.tensor([[0,1], [1,2]], dtype=torch.long)

        emb = model.get_multimodal_embeddings(
            x=x, edge_index=edge_index,
            text="Test text",
            user_idx=0,
            device="cpu",
        )

        assert "graph_embedding" in emb
        assert "text_embedding" in emb
        assert emb["graph_embedding"].shape == (16,)
        assert emb["text_embedding"].shape == (768,)


class TestGraphExtractor:
    """Test graph extraction (mock Neo4j)."""

    def test_knn_fallback(self):
        extractor = GraphExtractor(neo4j=None)

        user_df = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'],
            'followers': [100, 200, 300],
            'verified': [0, 1, 0],
            'post_count': [10, 20, 30],
            'avg_sentiment': [0.1, -0.2, 0.0],
            'avg_toxicity': [0.1, 0.3, 0.2],
            'controversy_rate': [0.0, 0.1, 0.05],
        })

        edge_index = extractor._knn_fallback(user_df, k=2)
        assert edge_index.shape[0] == 2  # [2, num_edges]
        assert edge_index.shape[1] > 0


class TestAdversarialGenerator:
    """Test adversarial generator with mock predictor."""

    def test_strategies(self):
        class MockPredictor:
            def predict(self, text, author_id):
                # Simple heuristic: longer text = higher doom
                return {
                    'probability': min(len(text) / 500, 0.99),
                    'doom_score': int(min(len(text) / 500, 0.99) * 100),
                    'risk_level': 'HIGH',
                }

        generator = AdversarialGenerator(MockPredictor())

        text = "This is a test post."
        variants = generator.generate_variants(text, max_variants=3, toxicity_budget=0.9)

        assert isinstance(variants, list)
        if variants:
            assert all(isinstance(v, AttackResult) for v in variants)
            assert all(v.doom_uplift >= 0 for v in variants)


class TestDataProcessing:
    """Test data processing utilities."""

    def test_proper_labels(self):
        df = pd.DataFrame({
            'text': [
                'This is a normal post about cats.',
                'This celebrity was fired after boycott petition.',
                'Just enjoying my day.',
                'Company apologized and removed the product after backlash.',
            ],
            'likes': [10, 1000, 5, 2000],
            'sentiment_polarity': [0.5, -0.8, 0.2, -0.6],
            'replies': [2, 500, 1, 800],
        })

        from train_model_full_fixed import create_proper_labels
        labels = create_proper_labels(df)

        assert len(labels) == 4
        assert labels.iloc[0] == 0  # Normal post
        assert labels.iloc[1] == 1  # Action keywords + high engagement
        assert labels.iloc[3] == 1  # Multiple signals


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
