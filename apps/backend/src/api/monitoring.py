"""Production monitoring and observability for Doom Index.

Features:
- Prometheus metrics export
- Request latency tracking
- Model drift detection
- Health checks with dependency status
- Structured logging with correlation IDs
"""

import logging
import time
from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try Prometheus
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not installed. Metrics disabled.")


class DoomMetrics:
    """Prometheus metrics collector for Doom Index.
    
    Tracks:
    - Request counts and latencies
    - Prediction distributions
    - Model performance over time
    - System health
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        
        if self.enabled:
            # Request metrics
            self.request_count = Counter(
                "doom_requests_total",
                "Total requests",
                ["endpoint", "status"]
            )
            self.request_latency = Histogram(
                "doom_request_duration_seconds",
                "Request latency",
                ["endpoint"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            )
            
            # Prediction metrics
            self.prediction_distribution = Histogram(
                "doom_prediction_probability",
                "Distribution of doom probabilities",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            )
            self.risk_level_count = Counter(
                "doom_risk_level_total",
                "Count of predictions by risk level",
                ["level"]
            )
            
            # Model metrics
            self.model_info = Info("doom_model", "Model information")
            self.model_latency = Histogram(
                "doom_model_inference_seconds",
                "Model inference time",
                ["model_component"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
            )
            
            # System metrics
            self.active_requests = Gauge("doom_active_requests", "Currently active requests")
            self.cache_hit_rate = Gauge("doom_cache_hit_rate", "Cache hit rate")
            
            logger.info("Prometheus metrics initialized")
    
    def record_request(self, endpoint: str, status: str, latency: float):
        """Record API request metrics."""
        if not self.enabled:
            return
        self.request_count.labels(endpoint=endpoint, status=status).inc()
        self.request_latency.labels(endpoint=endpoint).observe(latency)
    
    def record_prediction(self, probability: float, risk_level: str):
        """Record prediction metrics."""
        if not self.enabled:
            return
        self.prediction_distribution.observe(probability)
        self.risk_level_count.labels(level=risk_level).inc()
    
    def record_model_latency(self, component: str, latency: float):
        """Record model component latency."""
        if not self.enabled:
            return
        self.model_latency.labels(model_component=component).observe(latency)
    
    def set_model_info(self, version: str, architecture: str):
        """Set model metadata."""
        if not self.enabled:
            return
        self.model_info.info({"version": version, "architecture": architecture})
    
    def get_prometheus_metrics(self) -> bytes:
        """Get metrics in Prometheus exposition format."""
        if not self.enabled:
            return b"# metrics disabled\n"
        return generate_latest()
    
    @contextmanager
    def track_request(self, endpoint: str):
        """Context manager to track request lifecycle."""
        if self.enabled:
            self.active_requests.inc()
        start = time.time()
        try:
            yield
            status = "success"
        except Exception:
            status = "error"
            raise
        finally:
            latency = time.time() - start
            self.record_request(endpoint, status, latency)
            if self.enabled:
                self.active_requests.dec()


class DriftDetector:
    """Detect model drift by comparing prediction distributions over time.
    
    Uses KL divergence and PSI (Population Stability Index) to detect
    when incoming data distribution shifts from training distribution.
    """
    
    def __init__(
        self,
        reference_probs: Optional[np.ndarray] = None,
        psi_threshold: float = 0.25,
        kl_threshold: float = 0.1,
        window_size: int = 1000,
    ):
        self.reference_probs = reference_probs
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.window_size = window_size
        self.recent_probs = []
    
    def update(self, probabilities: np.ndarray):
        """Add new predictions to sliding window."""
        self.recent_probs.extend(probabilities.tolist())
        if len(self.recent_probs) > self.window_size:
            self.recent_probs = self.recent_probs[-self.window_size:]
    
    def compute_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index."""
        breakpoints = np.linspace(0, 1, bins + 1)
        
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Add small epsilon to avoid division by zero
        expected_percents = np.clip(expected_percents, 0.0001, 1.0)
        actual_percents = np.clip(actual_percents, 0.0001, 1.0)
        
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi)
    
    def compute_kl_divergence(self, expected: np.ndarray, actual: np.ndarray, bins: int = 20) -> float:
        """Compute KL divergence between distributions."""
        hist_expected, bin_edges = np.histogram(expected, bins=bins, range=(0, 1))
        hist_actual, _ = np.histogram(actual, bins=bin_edges)
        
        p = hist_expected / len(expected)
        q = hist_actual / len(actual)
        
        # Clip to avoid log(0)
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        
        kl = np.sum(p * np.log(p / q))
        return float(kl)
    
    def check_drift(self) -> Dict:
        """Check for model drift.
        
        Returns:
            Dict with drift status, PSI, KL divergence, and recommendation
        """
        if self.reference_probs is None or len(self.recent_probs) < 100:
            return {
                "drift_detected": False,
                "status": "insufficient_data",
                "psi": None,
                "kl_divergence": None,
                "recommendation": "Collect more predictions",
            }
        
        recent = np.array(self.recent_probs)
        
        psi = self.compute_psi(self.reference_probs, recent)
        kl = self.compute_kl_divergence(self.reference_probs, recent)
        
        drift_detected = psi > self.psi_threshold or kl > self.kl_threshold
        
        recommendation = "No action needed"
        if drift_detected:
            if psi > 0.3:
                recommendation = "URGENT: Retrain model immediately"
            elif psi > 0.25:
                recommendation = "WARNING: Schedule model retraining"
            else:
                recommendation = "Monitor closely"
        
        return {
            "drift_detected": drift_detected,
            "status": "drift" if drift_detected else "stable",
            "psi": round(psi, 4),
            "kl_divergence": round(kl, 4),
            "samples_analyzed": len(recent),
            "recommendation": recommendation,
        }
