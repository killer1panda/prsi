"""
Production drift detection: monitors feature distributions, prediction distributions,
and concept drift using statistical tests and learned detectors.
"""
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    reference_window_size: int = 1000
    detection_window_size: int = 500
    ks_threshold: float = 0.05  # p-value threshold for KS test
    psi_threshold: float = 0.25
    prediction_drift_threshold: float = 0.1
    feature_names: List[str] = field(default_factory=list)
    device: str = "cpu"  # Drift detection usually on CPU


class DriftDetector:
    """
    Multi-modal drift detection for production ML systems.
    Detects: data drift (feature distributions), concept drift (prediction quality),
    and covariate shift.
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self.reference_stats: Dict[str, Dict] = {}
        self.reference_predictions: deque = deque(maxlen=self.config.reference_window_size)
        self.detection_buffer: Dict[str, deque] = {}
        self.prediction_buffer: deque = deque(maxlen=self.config.detection_window_size)

        # Learned drift detector (autoencoder-based)
        self.autoencoder: Optional[nn.Module] = None
        self.ae_threshold: float = 0.0

        logger.info("DriftDetector initialized")

    def fit_reference(self, features: np.ndarray, predictions: Optional[np.ndarray] = None):
        """
        Compute reference statistics from training/validation data.

        Args:
            features: (N, D) reference feature matrix
            predictions: (N,) optional reference predictions
        """
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        # Per-feature statistics
        for i in range(features.shape[1]):
            feat_name = self.config.feature_names[i] if i < len(self.config.feature_names) else f"feat_{i}"
            self.reference_stats[feat_name] = {
                "mean": float(np.mean(features[:, i])),
                "std": float(np.std(features[:, i])),
                "min": float(np.min(features[:, i])),
                "max": float(np.max(features[:, i])),
                "hist": np.histogram(features[:, i], bins=50, density=True),
                "percentiles": np.percentile(features[:, i], [5, 25, 50, 75, 95]).tolist()
            }

        if predictions is not None:
            self.reference_predictions.extend(predictions.tolist())

        # Fit autoencoder for learned drift detection
        self._fit_autoencoder(features)

        logger.info(f"Reference fitted on {features.shape[0]} samples, {features.shape[1]} features")

    def _fit_autoencoder(self, features: np.ndarray):
        """Fit a simple autoencoder to learn reference distribution."""
        input_dim = features.shape[1]
        hidden_dim = max(input_dim // 2, 16)

        class SimpleAE(nn.Module):
            def __init__(self, inp, hid):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, hid // 2)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hid // 2, hid), nn.ReLU(), nn.Linear(hid, inp)
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        self.autoencoder = SimpleAE(input_dim, hidden_dim)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        X = torch.tensor(features, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X, X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        self.autoencoder.train()
        for epoch in range(20):
            for batch_x, _ in loader:
                optimizer.zero_grad()
                recon = self.autoencoder(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        # Set threshold as 95th percentile of reconstruction error
        self.autoencoder.eval()
        with torch.no_grad():
            recon = self.autoencoder(X)
            errors = torch.mean((X - recon) ** 2, dim=1).numpy()
            self.ae_threshold = float(np.percentile(errors, 95))

        logger.info(f"Autoencoder fitted. Reconstruction threshold: {self.ae_threshold:.6f}")

    def update(self, features: np.ndarray, predictions: Optional[np.ndarray] = None):
        """Add new observations to detection buffer."""
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        for i in range(features.shape[1]):
            feat_name = self.config.feature_names[i] if i < len(self.config.feature_names) else f"feat_{i}"
            if feat_name not in self.detection_buffer:
                self.detection_buffer[feat_name] = deque(maxlen=self.config.detection_window_size)
            self.detection_buffer[feat_name].extend(features[:, i].tolist())

        if predictions is not None:
            self.prediction_buffer.extend(predictions.tolist())

    def detect(self) -> Dict[str, Any]:
        """
        Run all drift detection tests.

        Returns:
            Dict with drift flags, p-values, and detailed metrics
        """
        results = {
            "drift_detected": False,
            "feature_drift": {},
            "prediction_drift": {},
            "autoencoder_drift": False,
            "overall_risk": "low"
        }

        drift_count = 0
        total_features = 0

        # 1. Feature drift: KS test per feature
        for feat_name, ref_stats in self.reference_stats.items():
            if feat_name not in self.detection_buffer or len(self.detection_buffer[feat_name]) < 100:
                continue

            ref_samples = np.random.normal(
                ref_stats["mean"], ref_stats["std"], 
                self.config.reference_window_size
            )
            det_samples = np.array(self.detection_buffer[feat_name])

            ks_stat, p_value = stats.ks_2samp(ref_samples, det_samples)
            drift = p_value < self.config.ks_threshold

            # PSI calculation
            psi = self._calculate_psi(ref_samples, det_samples)

            results["feature_drift"][feat_name] = {
                "drift": drift or psi > self.config.psi_threshold,
                "ks_stat": float(ks_stat),
                "p_value": float(p_value),
                "psi": float(psi)
            }

            if drift or psi > self.config.psi_threshold:
                drift_count += 1
            total_features += 1

        # 2. Prediction drift
        if len(self.reference_predictions) > 100 and len(self.prediction_buffer) > 100:
            ref_pred = np.array(self.reference_predictions)
            det_pred = np.array(self.prediction_buffer)

            ks_stat, p_value = stats.ks_2samp(ref_pred, det_pred)
            mean_shift = abs(np.mean(det_pred) - np.mean(ref_pred))

            results["prediction_drift"] = {
                "drift": p_value < self.config.ks_threshold or mean_shift > self.config.prediction_drift_threshold,
                "ks_stat": float(ks_stat),
                "p_value": float(p_value),
                "mean_shift": float(mean_shift)
            }

            if results["prediction_drift"]["drift"]:
                drift_count += 1

        # 3. Autoencoder drift
        if self.autoencoder is not None and len(self.detection_buffer) > 0:
            # Use first feature buffer as proxy for sample count
            first_buffer = list(self.detection_buffer.values())[0]
            if len(first_buffer) >= 100:
                # Reconstruct recent samples
                recent = np.array([
                    list(self.detection_buffer.get(f"feat_{i}", [0]*100))[-100:]
                    for i in range(len(self.reference_stats))
                ]).T

                if recent.shape[1] == len(self.reference_stats):
                    X_recent = torch.tensor(recent, dtype=torch.float32)
                    self.autoencoder.eval()
                    with torch.no_grad():
                        recon = self.autoencoder(X_recent)
                        errors = torch.mean((X_recent - recon) ** 2, dim=1).numpy()

                    ae_drift = float(np.mean(errors)) > self.ae_threshold * 1.5
                    results["autoencoder_drift"] = ae_drift
                    results["ae_mean_error"] = float(np.mean(errors))
                    if ae_drift:
                        drift_count += 1

        # Overall assessment
        if total_features > 0:
            drift_ratio = drift_count / (total_features + 2)  # +2 for pred and ae
            if drift_ratio > 0.3:
                results["overall_risk"] = "critical"
                results["drift_detected"] = True
            elif drift_ratio > 0.15:
                results["overall_risk"] = "high"
                results["drift_detected"] = True
            elif drift_ratio > 0.05:
                results["overall_risk"] = "medium"

        return results

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        expected_percents = np.histogram(expected, bins=bin_edges)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=bin_edges)[0] / len(actual)

        # Add small epsilon to avoid division by zero
        expected_percents = np.clip(expected_percents, 1e-10, 1.0)
        actual_percents = np.clip(actual_percents, 1e-10, 1.0)

        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi)

    def reset(self):
        """Reset detection buffers (but keep reference)."""
        self.detection_buffer.clear()
        self.prediction_buffer.clear()
        logger.info("Drift detection buffers reset")
