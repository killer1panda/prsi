"""Feature engineering for doom-index dataset.

This module processes the raw dataset to extract and engineer features
for machine learning models.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from .sentiment import analyze_text_sentiment
from .toxicity import analyze_text_toxicity

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering pipeline."""

    def __init__(self):
        pass

    def process_dataset(
        self,
        input_csv: str,
        output_csv: str = None,
        sample_size: Optional[int] = None,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """Process dataset to add sentiment and toxicity features.

        Args:
            input_csv: Path to input CSV
            output_csv: Path to output CSV (optional)
            sample_size: Number of samples to process (for testing)
            batch_size: Batch size for processing

        Returns:
            Processed DataFrame
        """
        logger.info(f"Loading dataset from {input_csv}")
        df = pd.read_csv(input_csv).reset_index(drop=True)

        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {len(df)} rows for processing")

        logger.info(f"Processing {len(df)} rows with batch size {batch_size}")

        # Add sentiment features
        sentiment_features = []
        toxicity_features = []

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")

            batch_sentiment = []
            batch_toxicity = []

            for _, row in batch.iterrows():
                text = str(row.get('text', ''))

                # Sentiment analysis
                sentiment = analyze_text_sentiment(text) or {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
                batch_sentiment.append(sentiment)

                # Toxicity analysis
                toxicity = analyze_text_toxicity(text) or {'toxicity': 0.0}
                batch_toxicity.append(toxicity)

            sentiment_features.extend(batch_sentiment)
            toxicity_features.extend(batch_toxicity)

        # Add to dataframe
        sentiment_df = pd.DataFrame(sentiment_features)
        toxicity_df = pd.DataFrame(toxicity_features)

        # Rename columns to avoid conflicts
        sentiment_df = sentiment_df.add_prefix('sentiment_')
        toxicity_df = toxicity_df.add_prefix('toxicity_')

        # Drop existing columns to avoid duplicates
        existing_cols = set(df.columns)
        new_cols = set(sentiment_df.columns) | set(toxicity_df.columns)
        cols_to_drop = existing_cols & new_cols
        if cols_to_drop:
            df = df.drop(columns=list(cols_to_drop), errors='ignore')

        df = pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True), toxicity_df.reset_index(drop=True)], axis=1)

        # Add engineered features
        df = self._add_engineered_features(df)

        if output_csv:
            logger.info(f"Saving processed dataset to {output_csv}")
            df.to_csv(output_csv, index=False)

        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df

    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the dataset."""
        df = df.reset_index(drop=True)

        # Text length features
        df['text_length'] = df['text'].fillna('').str.len()
        df['word_count'] = df['text'].fillna('').str.split().str.len()
        df['hashtag_count'] = df['hashtags'].fillna('').str.count('#')

        # Sentiment-based features
        df.loc[:, 'sentiment_polarity'] = df['sentiment_compound']
        df.loc[:, 'sentiment_intensity'] = df[['sentiment_pos', 'sentiment_neg']].max(axis=1)

        # Toxicity flags
        if 'toxicity_toxicity' in df.columns:
            df.loc[:, 'is_toxic'] = (df['toxicity_toxicity'] > 0.7).astype(int)
        else:
            df['is_toxic'] = 0

        # Time features
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.dayofweek

        # Keyword-based features
        cancellation_keywords = ['cancel', 'cancelled', 'backlash', 'controversy', 'boycott', 'outrage', 'petition']
        for kw in cancellation_keywords:
            df[f'has_{kw}'] = df['text'].fillna('').str.lower().str.contains(kw).astype(int)

        return df

    def create_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: list = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create feature matrix and target vector for ML.

        Args:
            df: Processed DataFrame
            feature_cols: List of feature columns to use

        Returns:
            X, y arrays
        """
        if feature_cols is None:
            # Default feature set
            feature_cols = [
                'likes', 'retweets', 'replies', 'quotes',
                'text_length', 'word_count', 'hashtag_count',
                'sentiment_polarity', 'sentiment_intensity',
                'toxicity_toxicity', 'is_toxic',
                'hour', 'day_of_week'
            ] + [f'has_{kw}' for kw in ['cancel', 'cancelled', 'backlash', 'controversy', 'boycott', 'outrage', 'petition']]

        # Filter to available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        logger.info(f"Using features: {available_cols}")

        X = df[available_cols].fillna(0).values

        # Target: whether it's a cancellation-related post (1) or not (0)
        # For now, use presence of cancellation keywords as proxy
        y = df[[f'has_{kw}' for kw in ['cancel', 'cancelled', 'backlash', 'controversy', 'boycott', 'outrage', 'petition']]].any(axis=1).astype(int).values

        return X, y


def process_dataset_for_ml(input_csv: str, output_csv: str = None, sample_size: int = None) -> pd.DataFrame:
    """Convenience function to process dataset for ML."""
    engineer = FeatureEngineer()
    return engineer.process_dataset(input_csv, output_csv, sample_size)