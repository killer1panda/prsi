"""Ensemble methods for Doom Index.

Combines multiple models for improved robustness:
- Weighted voting (RF + GNN + BERT)
- Stacking with meta-learner
- Uncertainty quantification via disagreement
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DoomEnsemble:
    """Ensemble of Doom Index models.
    
    Combines predictions from:
    1. RandomForest baseline (fast, tabular features)
    2. Multimodal GNN+BERT (deep, contextual)
    3. Optional: Text-only BERT (when graph unavailable)
    """
    
    def __init__(
        self,
        rf_predictor=None,
        multimodal_predictor=None,
        text_only_predictor=None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.rf_predictor = rf_predictor
        self.multimodal_predictor = multimodal_predictor
        self.text_only_predictor = text_only_predictor
        
        # Default weights based on validation performance
        self.weights = weights or {
            "rf": 0.2,
            "multimodal": 0.7,
            "text_only": 0.1,
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
    
    def predict(
        self,
        text: str,
        author_id: str = "anonymous",
        features: Optional[dict] = None,
    ) -> Dict:
        """Ensemble prediction with uncertainty quantification.
        
        Returns:
            Combined prediction + individual model predictions + uncertainty
        """
        predictions = {}
        
        # RandomForest
        if self.rf_predictor and features:
            try:
                rf_pred = self.rf_predictor.predict_proba([features])[0]
                predictions["rf"] = {"prob": rf_pred[1], "pred": int(rf_pred[1] > 0.5)}
            except Exception as e:
                logger.debug(f"RF prediction failed: {e}")
        
        # Multimodal
        if self.multimodal_predictor:
            try:
                mm_pred = self.multimodal_predictor.predict(text, author_id)
                predictions["multimodal"] = {
                    "prob": mm_pred["probability"],
                    "pred": mm_pred["prediction"],
                }
            except Exception as e:
                logger.debug(f"Multimodal prediction failed: {e}")
        
        # Text-only fallback
        if self.text_only_predictor:
            try:
                to_pred = self.text_only_predictor.predict(text, author_id)
                predictions["text_only"] = {
                    "prob": to_pred["probability"],
                    "pred": to_pred["prediction"],
                }
            except Exception as e:
                logger.debug(f"Text-only prediction failed: {e}")
        
        # Weighted ensemble
        if not predictions:
            raise RuntimeError("No models available for prediction")
        
        ensemble_prob = 0.0
        total_weight = 0.0
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0.0)
            ensemble_prob += weight * pred["prob"]
            total_weight += weight
        
        if total_weight > 0:
            ensemble_prob /= total_weight
        
        ensemble_pred = int(ensemble_prob > 0.5)
        
        # Uncertainty = variance across models
        probs = [p["prob"] for p in predictions.values()]
        uncertainty = np.var(probs) if len(probs) > 1 else 0.0
        
        # Disagreement = models don't agree
        preds = [p["pred"] for p in predictions.values()]
        disagreement = len(set(preds)) > 1
        
        return {
            "prediction": ensemble_pred,
            "probability": ensemble_prob,
            "doom_score": int(ensemble_prob * 100),
            "risk_level": self._get_risk_level(ensemble_prob),
            "uncertainty": float(uncertainty),
            "disagreement": disagreement,
            "individual_predictions": predictions,
            "explanation": self._generate_explanation(predictions, uncertainty, disagreement),
        }
    
    def _get_risk_level(self, prob: float) -> str:
        if prob > 0.7:
            return "CRITICAL"
        elif prob > 0.4:
            return "HIGH"
        elif prob > 0.2:
            return "MODERATE"
        return "LOW"
    
    def _generate_explanation(self, predictions, uncertainty, disagreement) -> str:
        parts = []
        
        if disagreement:
            parts.append("Models disagree on this prediction.")
        
        if uncertainty > 0.05:
            parts.append(f"High uncertainty ({uncertainty:.3f}). Consider manual review.")
        
        if not parts:
            parts.append("All models agree on this prediction.")
        
        return " ".join(parts)
    
    def calibrate_weights(self, val_data: List[Dict]):
        """Learn optimal ensemble weights from validation data.
        
        Uses a simple grid search to maximize validation F1.
        """
        best_f1 = 0.0
        best_weights = self.weights.copy()
        
        # Grid search over weight combinations
        for w_rf in np.arange(0, 1.1, 0.1):
            for w_mm in np.arange(0, 1.1 - w_rf, 0.1):
                w_to = 1.0 - w_rf - w_mm
                weights = {"rf": w_rf, "multimodal": w_mm, "text_only": w_to}
                
                # Evaluate on validation set
                f1 = self._evaluate_weights(weights, val_data)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = weights
        
        self.weights = best_weights
        logger.info(f"Calibrated weights: {self.weights} (F1: {best_f1:.4f})")
    
    def _evaluate_weights(self, weights, val_data):
        """Evaluate a weight configuration."""
        from sklearn.metrics import f1_score
        
        y_true = []
        y_pred = []
        
        for sample in val_data:
            # Temporarily set weights
            old_weights = self.weights
            self.weights = weights
            
            result = self.predict(sample["text"], sample.get("author_id", ""))
            
            self.weights = old_weights
            
            y_true.append(sample["label"])
            y_pred.append(result["prediction"])
        
        return f1_score(y_true, y_pred)
