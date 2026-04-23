"""Temporal dynamics and time-series forecasting for Doom Index.

Senior-level implementation:
- Temporal feature extraction (velocity, acceleration, trend)
- ARIMA/Prophet-style doom trajectory forecasting
- Temporal Graph Networks (TGN) concepts for time-aware GNN
- User activity timeline encoding with positional encoding
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor:
    """Extract time-series features from user post history.
    
    Computes velocity, acceleration, and trend features
    that capture how a user's behavior evolves over time.
    """
    
    def __init__(
        self,
        window_sizes: List[int] = [7, 14, 30],  # days
        sentiment_col: str = "sentiment_polarity",
        engagement_col: str = "likes",
    ):
        self.window_sizes = window_sizes
        self.sentiment_col = sentiment_col
        self.engagement_col = engagement_col
    
    def extract(self, user_posts: pd.DataFrame) -> Dict[str, float]:
        """Extract temporal features from a user's post history.
        
        Args:
            user_posts: DataFrame with user's posts, sorted by time
        
        Returns:
            Dictionary of temporal features
        """
        if len(user_posts) < 3:
            return self._default_features()
        
        features = {}
        
        # Ensure sorted by time
        posts = user_posts.sort_values("created_at")
        
        # 1. Sentiment velocity (rate of change)
        sentiment = posts[self.sentiment_col].fillna(0).values
        if len(sentiment) >= 2:
            features["sentiment_velocity"] = np.mean(np.diff(sentiment))
            features["sentiment_acceleration"] = np.mean(np.diff(np.diff(sentiment))) if len(sentiment) >= 3 else 0.0
        else:
            features["sentiment_velocity"] = 0.0
            features["sentiment_acceleration"] = 0.0
        
        # 2. Engagement trend (slope of linear fit)
        engagement = posts[self.engagement_col].fillna(0).values
        x = np.arange(len(engagement)).reshape(-1, 1)
        if len(engagement) >= 2:
            model = LinearRegression().fit(x, engagement)
            features["engagement_trend"] = model.coef_[0]
            features["engagement_trend_r2"] = model.score(x, engagement)
        else:
            features["engagement_trend"] = 0.0
            features["engagement_trend_r2"] = 0.0
        
        # 3. Window-based volatility
        for window in self.window_sizes:
            if len(sentiment) >= window:
                recent = sentiment[-window:]
                features[f"sentiment_volatility_{window}d"] = np.std(recent)
                features[f"sentiment_mean_{window}d"] = np.mean(recent)
            else:
                features[f"sentiment_volatility_{window}d"] = np.std(sentiment) if len(sentiment) > 1 else 0.0
                features[f"sentiment_mean_{window}d"] = np.mean(sentiment) if len(sentiment) > 0 else 0.0
        
        # 4. Posting frequency changes
        if "created_at" in posts.columns:
            timestamps = pd.to_datetime(posts["created_at"], unit="s", errors="coerce")
            if len(timestamps.dropna()) >= 2:
                intervals = timestamps.diff().dt.total_seconds().dropna()
                features["avg_posting_interval"] = intervals.mean()
                features["posting_interval_volatility"] = intervals.std() if len(intervals) > 1 else 0.0
                
                # Are posts accelerating? (shorter intervals)
                if len(intervals) >= 3:
                    x = np.arange(len(intervals)).reshape(-1, 1)
                    model = LinearRegression().fit(x, intervals.values)
                    features["posting_acceleration"] = -model.coef_[0]  # Negative slope = accelerating
                else:
                    features["posting_acceleration"] = 0.0
            else:
                features["avg_posting_interval"] = 86400.0
                features["posting_interval_volatility"] = 0.0
                features["posting_acceleration"] = 0.0
        
        # 5. Controversy escalation
        if "toxicity" in posts.columns:
            toxicity = posts["toxicity"].fillna(0).values
            features["toxicity_trend"] = np.mean(np.diff(toxicity)) if len(toxicity) >= 2 else 0.0
            features["max_toxicity_recent"] = np.max(toxicity[-7:]) if len(toxicity) >= 7 else np.max(toxicity)
        
        return features
    
    def _default_features(self) -> Dict[str, float]:
        """Default features for users with insufficient history."""
        defaults = {
            "sentiment_velocity": 0.0,
            "sentiment_acceleration": 0.0,
            "engagement_trend": 0.0,
            "engagement_trend_r2": 0.0,
            "sentiment_volatility_7d": 0.0,
            "sentiment_mean_7d": 0.0,
            "sentiment_volatility_14d": 0.0,
            "sentiment_mean_14d": 0.0,
            "sentiment_volatility_30d": 0.0,
            "sentiment_mean_30d": 0.0,
            "avg_posting_interval": 86400.0,
            "posting_interval_volatility": 0.0,
            "posting_acceleration": 0.0,
            "toxicity_trend": 0.0,
            "max_toxicity_recent": 0.0,
        }
        return defaults
    
    def extract_all_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features for all users in dataframe."""
        logger.info(f"Extracting temporal features for {df['author_id'].nunique()} users...")
        
        all_features = []
        for author_id, user_posts in df.groupby("author_id"):
            feats = self.extract(user_posts)
            feats["author_id"] = author_id
            all_features.append(feats)
        
        return pd.DataFrame(all_features)


class TemporalPositionalEncoding(nn.Module):
    """Learnable temporal positional encoding for user activity timelines.
    
    Encodes the relative time position of posts in a user's history,
    allowing the model to understand temporal ordering.
    """
    
    def __init__(self, d_model: int = 128, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embeddings
        self.pe = nn.Embedding(max_len, d_model)
        
        # Initialize with sinusoidal pattern (better than random)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe.weight.data = pe
        self.pe.weight.requires_grad = True  # Fine-tuneable
    
    def forward(self, x, timestamps):
        """
        Args:
            x: [batch, seq_len, d_model]
            timestamps: [batch, seq_len] — relative time indices (0 = most recent)
        """
        # Clamp timestamps to valid range
        timestamps = torch.clamp(timestamps, 0, self.pe.num_embeddings - 1)
        x = x + self.pe(timestamps)
        return self.dropout(x)


class TemporalDoomForecaster:
    """Forecast future doom trajectory using time-series methods.
    
    Uses a hybrid approach:
    - Short-term: Exponential smoothing
    - Medium-term: Linear trend projection
    - Long-term: Prophet-style seasonality (if enough data)
    """
    
    def __init__(self, horizon_days: int = 7):
        self.horizon_days = horizon_days
    
    def forecast(self, historical_doom_scores: List[float]) -> Dict:
        """Forecast doom trajectory.
        
        Args:
            historical_doom_scores: List of daily doom scores (most recent last)
        
        Returns:
            Dictionary with forecast and confidence intervals
        """
        if len(historical_doom_scores) < 3:
            return {
                "forecast": [historical_doom_scores[-1]] * self.horizon_days if historical_doom_scores else [50.0] * self.horizon_days,
                "trend": "insufficient_data",
                "peak_day": None,
                "confidence": 0.0,
            }
        
        scores = np.array(historical_doom_scores)
        
        # Exponential smoothing (short-term)
        alpha = 0.3
        smoothed = scores[0]
        for s in scores[1:]:
            smoothed = alpha * s + (1 - alpha) * smoothed
        
        # Linear trend
        x = np.arange(len(scores))
        model = LinearRegression().fit(x.reshape(-1, 1), scores)
        trend_slope = model.coef_[0]
        
        # Forecast
        forecast = []
        for i in range(1, self.horizon_days + 1):
            trend_component = trend_slope * (len(scores) + i - 1) + model.intercept_
            exp_component = smoothed + trend_slope * i
            # Weighted combination
            pred = 0.6 * exp_component + 0.4 * trend_component
            forecast.append(float(np.clip(pred, 0, 100)))
        
        # Determine trend direction
        if trend_slope > 2:
            trend = "accelerating"
        elif trend_slope > 0.5:
            trend = "increasing"
        elif trend_slope < -2:
            trend = "decelerating"
        elif trend_slope < -0.5:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Peak prediction
        peak_day = np.argmax(forecast) + 1 if max(forecast) > scores[-1] else None
        
        # Confidence based on data length and variance
        confidence = min(len(scores) / 30, 1.0) * (1.0 - np.std(scores) / 50.0)
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "forecast": forecast,
            "trend": trend,
            "peak_day": int(peak_day) if peak_day else None,
            "confidence": float(confidence),
            "current_score": float(scores[-1]),
            "predicted_max": float(max(forecast)),
        }


class UserTimelineEncoder(nn.Module):
    """Encode a user's post timeline with temporal awareness.
    
    Uses a transformer with temporal positional encoding to process
    a sequence of a user's posts and produce a user representation.
    """
    
    def __init__(
        self,
        post_feature_dim: int = 6,  # sentiment, toxicity, likes, etc.
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_history: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.post_proj = nn.Linear(post_feature_dim, hidden_dim)
        self.temporal_pe = TemporalPositionalEncoding(hidden_dim, max_len=max_history)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, post_features, timestamps, mask=None):
        """
        Args:
            post_features: [batch, seq_len, post_feature_dim]
            timestamps: [batch, seq_len] — relative time positions
            mask: [batch, seq_len] — padding mask
        
        Returns:
            user_embedding: [batch, hidden_dim]
        """
        x = self.post_proj(post_features)  # [B, S, H]
        x = self.temporal_pe(x, timestamps)  # Add temporal encoding
        
        if mask is not None:
            # Invert mask for transformer (True = mask)
            mask = ~mask.bool()
        
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, S, H]
        
        # Mean pooling over valid positions
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        return self.output_proj(x)
