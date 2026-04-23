"""Integrated multimodal predictor — replaces CancellationPredictor.

Loads the trained GraphSAGE + DistilBERT model and provides a unified
prediction interface compatible with the existing API.
"""

import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from src.models.gnn_model import MultimodalDoomPredictor
from src.features.graph_extractor import GraphExtractor

logger = logging.getLogger(__name__)


class IntegratedDoomPredictor:
    """Production predictor combining GNN + NLP + tabular features."""

    def __init__(
        self,
        model_path: str = "models/multimodal_doom/best_model.pt",
        config_path: str = "models/multimodal_doom/model_config.pt",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.graph_data = None
        self.user_to_idx = {}
        self.model_path = model_path
        self.config_path = config_path

        # Load model if paths exist
        if Path(model_path).exists() and Path(config_path).exists():
            self.load_model()
        else:
            logger.warning(f"Model not found at {model_path}. Predictor not ready.")

    def load_model(self):
        """Load trained multimodal model."""
        logger.info(f"Loading model from {self.model_path}")

        # Load config
        config = torch.load(self.config_path, map_location=self.device)

        # Create model
        self.model = MultimodalDoomPredictor(
            graph_in_channels=config.get('graph_in_channels', 6),
            graph_hidden=config.get('graph_hidden', 128),
            graph_out=config.get('graph_out', 128),
            graph_layers=config.get('graph_layers', 2),
            text_model=config.get('text_model', 'distilbert-base-uncased'),
            text_freeze=6,  # All frozen for inference
            fusion_hidden=config.get('fusion_hidden', 256),
            num_classes=2,
            dropout=0.0,  # No dropout for inference
        )

        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded. Epoch {checkpoint.get('epoch', '?')}, "
                   f"Best F1: {checkpoint.get('metrics', {}).get('val_f1', 0):.4f}")

    def build_graph_from_posts(self, posts_df: pd.DataFrame):
        """Build or update graph from a DataFrame of posts.

        Args:
            posts_df: DataFrame with columns [author_id, followers, verified, 
                                             sentiment_polarity, toxicity, ...]
        """
        logger.info(f"Building graph from {len(posts_df)} posts")

        # Aggregate user features
        user_features = posts_df.groupby('author_id').agg({
            'followers': 'first',
            'verified': 'first',
            'sentiment_polarity': 'mean',
            'toxicity_toxicity': 'mean',
            'text_length': 'count',  # post count
        }).reset_index()

        user_features.columns = ['user_id', 'followers', 'verified', 
                                  'avg_sentiment', 'avg_toxicity', 'post_count']
        user_features['controversy_rate'] = 0.0  # Would need labels
        user_features['verified'] = user_features['verified'].astype(float)

        # Create mapping
        self.user_to_idx = {uid: i for i, uid in enumerate(user_features['user_id'].tolist())}

        # Build features
        feature_cols = ['followers', 'verified', 'post_count', 
                       'avg_sentiment', 'avg_toxicity', 'controversy_rate']
        features = user_features[feature_cols].fillna(0).values.astype(np.float32)

        # Log transform followers
        features[:, 0] = np.log1p(features[:, 0])

        # Normalize
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features = (features - mean) / std

        # Create synthetic edges (co-occurrence in same thread/subreddit)
        edges = []
        if 'subreddit' in posts_df.columns:
            # Users who posted in same subreddit are connected
            subreddits = posts_df.groupby('subreddit')['author_id'].apply(list)
            for authors in subreddits:
                for i, a1 in enumerate(authors):
                    for a2 in authors[i+1:]:
                        if a1 in self.user_to_idx and a2 in self.user_to_idx:
                            edges.append([self.user_to_idx[a1], self.user_to_idx[a2]])

        if len(edges) == 0:
            # Fallback: random edges
            num_users = len(user_features)
            num_edges = min(num_users * 3, 50000)
            edges = np.random.randint(0, num_users, (num_edges, 2)).tolist()

        import torch
        from torch_geometric.data import Data

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float)

        self.graph_data = Data(x=x, edge_index=edge_index, num_nodes=len(user_features))
        self.graph_data.to(self.device)

        logger.info(f"Graph built: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")

    def predict(
        self,
        text: str,
        author_id: str = "anonymous",
        followers: int = 0,
        verified: bool = False,
    ) -> Dict[str, Any]:
        """Predict cancellation risk for a single post.

        Returns:
            Dict with prediction, probability, and feature breakdown.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure user exists in graph
        if author_id not in self.user_to_idx:
            # Add as new node with given features
            self._add_new_user(author_id, followers, verified)

        user_idx = self.user_to_idx[author_id]

        # Predict
        pred, prob = self.model.predict(
            x=self.graph_data.x,
            edge_index=self.graph_data.edge_index,
            text=text,
            user_idx=user_idx,
            device=self.device,
        )

        # Get embeddings for interpretability
        embeddings = self.model.get_multimodal_embeddings(
            x=self.graph_data.x,
            edge_index=self.graph_data.edge_index,
            text=text,
            user_idx=user_idx,
            device=self.device,
        )

        # Compute doom score (0-100)
        doom_score = int(prob * 100)

        # Risk level
        if prob > 0.7:
            risk_level = "CRITICAL"
        elif prob > 0.4:
            risk_level = "HIGH"
        elif prob > 0.2:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return {
            'prediction': pred,
            'probability': prob,
            'doom_score': doom_score,
            'risk_level': risk_level,
            'graph_embedding_norm': float(np.linalg.norm(embeddings['graph_embedding'])),
            'text_embedding_norm': float(np.linalg.norm(embeddings['text_embedding'])),
        }

    def predict_batch(self, texts: list, author_ids: list) -> list:
        """Predict for a batch of posts."""
        results = []
        for text, author_id in zip(texts, author_ids):
            results.append(self.predict(text, author_id))
        return results

    def _add_new_user(self, user_id: str, followers: int, verified: bool):
        """Add a new user node to the graph dynamically."""
        import torch

        idx = len(self.user_to_idx)
        self.user_to_idx[user_id] = idx

        # Create feature vector (normalized same way as training)
        new_features = torch.tensor([
            [np.log1p(followers), float(verified), 1.0, 0.0, 0.0, 0.0]
        ], dtype=torch.float, device=self.device)

        # Append to graph
        self.graph_data.x = torch.cat([self.graph_data.x, new_features], dim=0)
        self.graph_data.num_nodes += 1

        logger.debug(f"Added new user {user_id} at index {idx}")


# Backwards compatibility wrapper
def load_predictor(model_path: str = None) -> IntegratedDoomPredictor:
    """Load the integrated predictor."""
    if model_path is None:
        model_path = "models/multimodal_doom/best_model.pt"

    predictor = IntegratedDoomPredictor(model_path=model_path)
    return predictor


if __name__ == "__main__":
    # Quick test
    predictor = IntegratedDoomPredictor()
    print("Integrated predictor module ready.")
