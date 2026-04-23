"""Model calibration for Doom Index.

Implements:
- Platt scaling (sigmoid calibration)
- Isotonic regression
- Temperature scaling
- Expected Calibration Error (ECE) computation
"""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Temperature scaling for model calibration.
    
    Learns a single temperature parameter T to soften
    the softmax distribution: p_i = exp(z_i/T) / sum(exp(z_j/T))
    
    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        """Scale logits by temperature."""
        return logits / self.temperature
    
    def fit(self, logits, labels, lr=0.01, max_iter=1000):
        """Fit temperature on validation set using NLL."""
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled = self.forward(logits_t)
            loss = nn.functional.cross_entropy(scaled, labels_t)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        logger.info(f"Temperature fitted: T = {self.temperature.item():.4f}")
        return self
    
    def calibrate(self, logits):
        """Apply temperature scaling to logits."""
        self.eval()
        with torch.no_grad():
            scaled = self.forward(torch.tensor(logits, dtype=torch.float32))
            probs = torch.softmax(scaled, dim=-1)
        return probs.numpy()


class PlattScaler:
    """Platt scaling using logistic regression.
    
    Fits a sigmoid to the model's confidence scores.
    """
    
    def __init__(self):
        self.model = LogisticRegression()
    
    def fit(self, probs, labels):
        """Fit Platt scaling.
        
        Args:
            probs: [N] positive class probabilities
            labels: [N] true labels (0 or 1)
        """
        # Use logit of probability as feature
        logits = np.log(np.clip(probs, 1e-10, 1 - 1e-10)) - np.log(np.clip(1 - probs, 1e-10, 1 - 1e-10))
        X = logits.reshape(-1, 1)
        self.model.fit(X, labels)
        logger.info("Platt scaling fitted")
    
    def calibrate(self, probs):
        """Apply Platt scaling."""
        logits = np.log(np.clip(probs, 1e-10, 1 - 1e-10)) - np.log(np.clip(1 - probs, 1e-10, 1 - 1e-10))
        X = logits.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


class CalibrationAnalyzer:
    """Analyze and improve model calibration."""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.temperature_scaler = None
        self.platt_scaler = None
        self.isotonic = None
    
    def compute_ece(self, probs, labels) -> float:
        """Compute Expected Calibration Error.
        
        ECE = sum_{m=1}^M (|B_m|/n) |acc(B_m) - conf(B_m)|
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > lower) & (probs <= upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def compute_mce(self, probs, labels) -> float:
        """Compute Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > lower) & (probs <= upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return float(mce)
    
    def fit_all(self, val_logits, val_labels):
        """Fit all calibration methods on validation data."""
        # Temperature scaling
        self.temperature_scaler = TemperatureScaler()
        self.temperature_scaler.fit(val_logits, val_labels)
        
        # Platt scaling
        val_probs = torch.softmax(torch.tensor(val_logits), dim=-1)[:, 1].numpy()
        self.platt_scaler = PlattScaler()
        self.platt_scaler.fit(val_probs, val_labels)
        
        # Isotonic regression
        self.isotonic = IsotonicRegression(out_of_bounds="clip")
        self.isotonic.fit(val_probs, val_labels)
        
        logger.info("All calibration methods fitted")
    
    def calibrate(self, logits, method: str = "temperature") -> np.ndarray:
        """Calibrate logits using specified method."""
        if method == "temperature" and self.temperature_scaler:
            return self.temperature_scaler.calibrate(logits)[:, 1]
        elif method == "platt" and self.platt_scaler:
            probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
            return self.platt_scaler.calibrate(probs)
        elif method == "isotonic" and self.isotonic:
            probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
            return self.isotonic.predict(probs)
        else:
            # No calibration
            return torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    
    def evaluate(self, logits, labels) -> Dict:
        """Evaluate calibration of all methods."""
        results = {}
        
        for method in ["none", "temperature", "platt", "isotonic"]:
            if method == "none":
                probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
            else:
                probs = self.calibrate(logits, method)
            
            ece = self.compute_ece(probs, labels)
            mce = self.compute_mce(probs, labels)
            
            results[method] = {
                "ece": round(ece, 4),
                "mce": round(mce, 4),
            }
        
        return results
