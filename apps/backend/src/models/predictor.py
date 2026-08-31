"""Machine learning models for doom-index.

This module provides training and prediction capabilities for
cancellation event detection and analysis.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.features import process_dataset_for_ml

logger = logging.getLogger(__name__)

class CancellationPredictor:
    """ML model for predicting cancellation events."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None

    def train(
        self,
        input_csv: str,
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            input_csv: Path to processed CSV
            model_type: Type of model ('random_forest', 'logistic', 'svm')
            test_size: Test set size
            sample_size: Sample size for training

        Returns:
            Training results
        """
        logger.info(f"Training {model_type} model on {input_csv}")

        # Process dataset
        from src.features import FeatureEngineer
        engineer = FeatureEngineer()
        df = engineer.process_dataset(input_csv, sample_size=sample_size)
        X, y = engineer.create_feature_matrix(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Select model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        elif model_type == 'svm':
            self.model = SVC(
                random_state=42,
                class_weight='balanced',
                probability=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        results = {
            'model_type': model_type,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'cross_val_scores': cross_val_score(self.model, X_train_scaled, y_train, cv=5).tolist(),
            'feature_importance': getattr(self.model, 'feature_importances_', None)
        }

        logger.info(f"Training complete. CV scores: {results['cross_val_scores']}")
        return results

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Predict cancellation likelihood.

        Args:
            features: Feature array

        Returns:
            Prediction (0/1), probability
        """
        if self.model is None:
            raise ValueError("Model not trained")

        if self.scaler:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        pred = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0, 1]

        return pred, proba

    def save_model(self, path: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_cols = model_data.get('feature_cols')

        logger.info(f"Model loaded from {path}")


def train_cancellation_model(
    input_csv: str,
    model_path: str = 'models/cancellation_predictor.pkl',
    model_type: str = 'random_forest'
) -> CancellationPredictor:
    """Train and save cancellation prediction model."""
    predictor = CancellationPredictor()
    results = predictor.train(input_csv, model_type=model_type)

    # Save model
    Path(model_path).parent.mkdir(exist_ok=True)
    predictor.save_model(model_path)

    return predictor, results