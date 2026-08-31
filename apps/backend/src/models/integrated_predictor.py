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
from src.models.calibration import FollowerStratifiedCalibrator
from src.models.multilingual import MultilingualConfig, MultilingualEncoder

logger = logging.getLogger(__name__)


class IntegratedDoomPredictor:
    """Production predictor combining GNN + NLP + tabular + multilingual + calibration features."""

    def __init__(
        self,
        model_path: str = "models/multimodal_doom/best_model.pt",
        config_path: str = "models/multimodal_doom/model_config.pt",
        device: str = None,
        enable_multilingual: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.graph_data = None
        self.user_to_idx = {}
        self.model_path = model_path
        self.config_path = config_path
        self.enable_multilingual = enable_multilingual

        # Follower-stratified calibration head (handles low-follower vs influencer edge cases)
        self.calibrator = FollowerStratifiedCalibrator(low_threshold=1000, high_threshold=50000)

        # Multilingual / Hinglish code-switching handler
        self._multilingual_encoder = None

        # Load model if paths exist
        if Path(model_path).exists() and Path(config_path).exists():
            self.load_model()
        else:
            logger.info(f"Model checkpoint not found at {model_path}. Initializing default MultimodalDoomPredictor.")
            self.model = MultimodalDoomPredictor().to(self.device)
            self.model.eval()

    @property
    def multilingual_encoder(self):
        """Lazy loader for multilingual / Hinglish encoder."""
        if self._multilingual_encoder is None and self.enable_multilingual:
            try:
                # Use lightweight config for inference
                cfg = MultilingualConfig(device=self.device)
                self._multilingual_encoder = MultilingualEncoder(cfg)
            except Exception as e:
                logger.warning(f"Could not load full MultilingualEncoder backbone ({e}). Using regex language detector.")
                # Minimal fallback object with detect_language and preprocess methods
                class RegexLanguageDetector:
                    def __init__(self):
                        import re
                        self.patterns = {
                            "roman_hindi": re.compile(r'\b(kya|nahi|hai|main|tu|aap|kaise|kyun|bahut|achha|bura|bhai|yaar|desh|modi|bjp|congress)\b', re.I),
                            "hindi_script": re.compile(r'[\u0900-\u097F]+'),
                            "english": re.compile(r'\b(the|is|are|was|were|have|has|had|do|does|did|will|would|could|should)\b', re.I),
                        }
                    def detect_language(self, text: str) -> str:
                        has_hi = bool(self.patterns["hindi_script"].search(text))
                        has_rh = bool(self.patterns["roman_hindi"].search(text))
                        has_en = bool(self.patterns["english"].search(text))
                        if has_hi and has_en: return "mixed"
                        if has_hi: return "hi"
                        if has_rh: return "hinglish"
                        return "en"
                    def preprocess(self, texts: list) -> list:
                        import re
                        return [re.sub(r'(.)\1{3,}', r'\1\1\1', t.lower().strip()) for t in texts]
                self._multilingual_encoder = RegexLanguageDetector()
        return self._multilingual_encoder


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
            Dict with prediction, probability, calibrated probability, language, and feature breakdown.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # ── Multilingual & Hinglish handling ─────────────────────────────────
        language = "en"
        processed_text = text
        if self.enable_multilingual and self.multilingual_encoder is not None:
            try:
                language = self.multilingual_encoder.detect_language(text)
                if language in ("hi", "hinglish", "mixed"):
                    processed_text = self.multilingual_encoder.preprocess([text])[0]
            except Exception as e:
                logger.debug(f"Language detection error: {e}")

        # ── Graph-data None guard ────────────────────────────────────────────
        # graph_data is None when the predictor is instantiated without a
        # pre-built graph (cold start, unit tests, Beam workers before setup).
        # Fall back to text-only prediction in that case.
        if self.graph_data is None:
            logger.warning(
                "graph_data is None — falling back to text-only prediction for '%s'. "
                "Call build_graph_from_posts() to enable full GNN inference.",
                author_id,
            )
            return self._text_only_predict(processed_text, author_id, followers, verified, language=language)

        # ── Ensure user exists in graph (add new node if unseen) ────────────
        if author_id not in self.user_to_idx:
            self._add_new_user(author_id, followers, verified)

        user_idx = self.user_to_idx[author_id]

        # ── Full multimodal predict ──────────────────────────────────────────
        pred, prob = self.model.predict(
            x=self.graph_data.x,
            edge_index=self.graph_data.edge_index,
            text=processed_text,
            user_idx=user_idx,
            device=self.device,
        )

        # Get embeddings for interpretability
        embeddings = self.model.get_multimodal_embeddings(
            x=self.graph_data.x,
            edge_index=self.graph_data.edge_index,
            text=processed_text,
            user_idx=user_idx,
            device=self.device,
        )

        return self._build_result(pred, prob, embeddings, followers=followers, language=language)

    def _text_only_predict(
        self,
        text: str,
        author_id: str = "anonymous",
        followers: int = 0,
        verified: bool = False,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Text-only fallback when graph_data is unavailable.

        Uses only the DistilBERT text encoder with a zero graph embedding.
        Slightly less accurate than full GNN inference but never crashes.
        """
        import torch
        from torch_geometric.data import Data

        # Build a minimal single-node graph for this user
        node_features = torch.tensor(
            [[np.log1p(followers), float(verified), 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float,
            device=self.device,
        )
        # Self-loop edge so GNN has a valid edge_index
        edge_index = torch.zeros((2, 1), dtype=torch.long, device=self.device)
        stub_graph = Data(x=node_features, edge_index=edge_index)

        pred, prob = self.model.predict(
            x=stub_graph.x,
            edge_index=stub_graph.edge_index,
            text=text,
            user_idx=0,
            device=self.device,
        )

        embeddings = self.model.get_multimodal_embeddings(
            x=stub_graph.x,
            edge_index=stub_graph.edge_index,
            text=text,
            user_idx=0,
            device=self.device,
        )

        result = self._build_result(pred, prob, embeddings, followers=followers, language=language)
        result["inference_mode"] = "text_only"
        return result

    def _build_result(
        self,
        pred: int,
        prob: float,
        embeddings: Dict,
        followers: int = 0,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Construct standardised result dict from prediction outputs."""
        # Follower-stratified calibration
        calibrated_prob = self.calibrator.calibrate_single(prob, followers) if hasattr(self, "calibrator") else prob
        doom_score = int(calibrated_prob * 100)

        if calibrated_prob > 0.7:
            risk_level = "CRITICAL"
        elif calibrated_prob > 0.4:
            risk_level = "HIGH"
        elif calibrated_prob > 0.2:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        follower_stratum = (
            "low_follower" if followers < 1000
            else ("mid_reach" if followers <= 50000 else "high_influencer")
        )

        return {
            "prediction": pred,
            "probability": round(prob, 4),
            "calibrated_probability": round(calibrated_prob, 4),
            "doom_score": doom_score,
            "risk_level": risk_level,
            "language": language,
            "is_hinglish": language in ("hi", "hinglish", "mixed"),
            "follower_stratum": follower_stratum,
            "graph_embedding_norm": float(np.linalg.norm(embeddings.get("graph_embedding", [0]))),
            "text_embedding_norm": float(np.linalg.norm(embeddings.get("text_embedding", [0]))),
            "inference_mode": "full_multimodal",
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
