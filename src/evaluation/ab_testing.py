"""
A/B testing framework for comparing model versions in production.
Implements statistical testing, traffic splitting, and automatic rollback.
"""
import logging
import hashlib
import time
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    test_name: str = "model_v2_vs_v1"
    control_model: str = "v1.0.0"
    treatment_model: str = "v2.0.0"
    traffic_split: float = 0.1  # 10% to treatment
    min_sample_size: int = 1000
    significance_level: float = 0.05
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = None
    max_duration_days: int = 14
    auto_rollback: bool = True
    rollback_threshold: float = 0.05  # Rollback if treatment worse by 5%


class TrafficRouter:
    """
    Routes requests to control or treatment based on consistent hashing.
    Ensures same user always hits same model version.
    """

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.split_point = int(65535 * config.traffic_split)

    def route(self, user_id: str) -> str:
        """
        Determine which model version to serve for a user.

        Args:
            user_id: Unique user identifier
        Returns:
            "control" or "treatment"
        """
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 65535
        return "treatment" if hash_val < self.split_point else "control"

    def get_model_version(self, user_id: str) -> str:
        """Get the actual model version string for a user."""
        bucket = self.route(user_id)
        return (
            self.config.treatment_model if bucket == "treatment" 
            else self.config.control_model
        )


class MetricsCollector:
    """Collect and store metrics for A/B test analysis."""

    def __init__(self):
        self.control_metrics: Dict[str, List[float]] = defaultdict(list)
        self.treatment_metrics: Dict[str, List[float]] = defaultdict(list)
        self.metadata: List[Dict] = []

    def record(self, user_id: str, bucket: str, metrics: Dict[str, float],
               timestamp: Optional[datetime] = None):
        """Record a single observation."""
        ts = timestamp or datetime.utcnow()

        target = self.control_metrics if bucket == "control" else self.treatment_metrics
        for metric_name, value in metrics.items():
            target[metric_name].append(value)

        self.metadata.append({
            "user_id": user_id,
            "bucket": bucket,
            "timestamp": ts.isoformat(),
            "metrics": metrics
        })

    def get_summary(self) -> Dict[str, Dict]:
        """Get statistical summary of collected metrics."""
        summary = {}
        all_metrics = set(self.control_metrics.keys()) | set(self.treatment_metrics.keys())

        for metric in all_metrics:
            control_vals = np.array(self.control_metrics.get(metric, []))
            treatment_vals = np.array(self.treatment_metrics.get(metric, []))

            summary[metric] = {
                "control": {
                    "n": len(control_vals),
                    "mean": float(np.mean(control_vals)) if len(control_vals) > 0 else 0,
                    "std": float(np.std(control_vals)) if len(control_vals) > 0 else 0,
                    "median": float(np.median(control_vals)) if len(control_vals) > 0 else 0
                },
                "treatment": {
                    "n": len(treatment_vals),
                    "mean": float(np.mean(treatment_vals)) if len(treatment_vals) > 0 else 0,
                    "std": float(np.std(treatment_vals)) if len(treatment_vals) > 0 else 0,
                    "median": float(np.median(treatment_vals)) if len(treatment_vals) > 0 else 0
                }
            }

        return summary


class StatisticalTester:
    """Run statistical tests for A/B test evaluation."""

    def __init__(self, config: ABTestConfig):
        self.config = config

    def t_test(self, control: np.ndarray, treatment: np.ndarray) -> Dict[str, float]:
        """Two-sample t-test for means."""
        if len(control) < 2 or len(treatment) < 2:
            return {"statistic": 0.0, "p_value": 1.0, "significant": False}

        statistic, p_value = stats.ttest_ind(control, treatment, equal_var=False)

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.config.significance_level,
            "control_mean": float(np.mean(control)),
            "treatment_mean": float(np.mean(treatment)),
            "difference": float(np.mean(treatment) - np.mean(control)),
            "relative_lift": float((np.mean(treatment) - np.mean(control)) / abs(np.mean(control))) if np.mean(control) != 0 else 0.0
        }

    def mann_whitney(self, control: np.ndarray, treatment: np.ndarray) -> Dict[str, float]:
        """Mann-Whitney U test (non-parametric alternative)."""
        if len(control) < 2 or len(treatment) < 2:
            return {"statistic": 0.0, "p_value": 1.0, "significant": False}

        statistic, p_value = stats.mannwhitneyu(control, treatment, alternative="two-sided")

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.config.significance_level
        }

    def bootstrap_ci(self, control: np.ndarray, treatment: np.ndarray,
                     n_bootstrap: int = 10000, ci: float = 0.95) -> Dict[str, float]:
        """Bootstrap confidence interval for difference in means."""
        if len(control) < 2 or len(treatment) < 2:
            return {"lower": 0.0, "upper": 0.0, "includes_zero": True}

        boot_diffs = []
        for _ in range(n_bootstrap):
            c_sample = np.random.choice(control, size=len(control), replace=True)
            t_sample = np.random.choice(treatment, size=len(treatment), replace=True)
            boot_diffs.append(np.mean(t_sample) - np.mean(c_sample))

        lower = np.percentile(boot_diffs, (1 - ci) / 2 * 100)
        upper = np.percentile(boot_diffs, (1 + ci) / 2 * 100)

        return {
            "lower": float(lower),
            "upper": float(upper),
            "includes_zero": lower <= 0 <= upper,
            "mean_difference": float(np.mean(boot_diffs))
        }

    def sequential_test(self, control: np.ndarray, treatment: np.ndarray,
                        max_samples: int = 10000) -> Dict[str, any]:
        """
        Sequential probability ratio test (SPRT) for early stopping.
        Stops when sufficient evidence accumulated.
        """
        # Simplified SPRT implementation
        for n in range(100, min(len(control), len(treatment), max_samples), 100):
            c_slice = control[:n]
            t_slice = treatment[:n]
            result = self.t_test(c_slice, t_slice)

            if result["p_value"] < self.config.significance_level:
                return {
                    "stopped_early": True,
                    "samples_used": n,
                    "result": result
                }

        return {
            "stopped_early": False,
            "samples_used": min(len(control), len(treatment)),
            "result": self.t_test(control, treatment)
        }


class ABTestRunner:
    """
    Orchestrates complete A/B test: routing, collection, analysis, decision.
    """

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.router = TrafficRouter(config)
        self.collector = MetricsCollector()
        self.tester = StatisticalTester(config)
        self.start_time = datetime.utcnow()
        self.status = "running"  # running, stopped, rolled_back, promoted
        logger.info(f"A/B test started: {config.test_name}")

    def route_request(self, user_id: str) -> str:
        """Route incoming request to appropriate model."""
        if self.status != "running":
            return "control"  # Default to control if test not running
        return self.router.route(user_id)

    def record_outcome(self, user_id: str, bucket: str, metrics: Dict[str, float]):
        """Record prediction outcome for analysis."""
        self.collector.record(user_id, bucket, metrics)

    def check_early_stopping(self) -> Optional[str]:
        """
        Check if test should stop early.

        Returns:
            "promote", "rollback", or None
        """
        summary = self.collector.get_summary()
        primary = self.config.primary_metric

        if primary not in summary:
            return None

        control_n = summary[primary]["control"]["n"]
        treatment_n = summary[primary]["treatment"]["n"]

        if control_n < self.config.min_sample_size or treatment_n < self.config.min_sample_size:
            return None

        # Run statistical test
        control_vals = np.array(self.collector.control_metrics[primary])
        treatment_vals = np.array(self.collector.treatment_metrics[primary])

        result = self.tester.t_test(control_vals, treatment_vals)

        # Auto-rollback check
        if self.config.auto_rollback and result["relative_lift"] < -self.config.rollback_threshold:
            self.status = "rolled_back"
            logger.warning(f"Auto-rollback triggered: treatment underperforming by {abs(result['relative_lift']):.2%}")
            return "rollback"

        # Check significance
        if result["significant"]:
            if result["relative_lift"] > 0:
                self.status = "stopped"
                logger.info(f"Treatment significant winner: +{result['relative_lift']:.2%}")
                return "promote"
            elif result["relative_lift"] < 0:
                self.status = "stopped"
                logger.info(f"Control significant winner: treatment underperforming")
                return "rollback"

        # Check max duration
        elapsed = datetime.utcnow() - self.start_time
        if elapsed.days >= self.config.max_duration_days:
            self.status = "stopped"
            logger.info("Max test duration reached")
            return "rollback" if result["relative_lift"] < 0 else "promote"

        return None

    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive A/B test report."""
        summary = self.collector.get_summary()

        statistical_results = {}
        for metric in summary.keys():
            control_vals = np.array(self.collector.control_metrics.get(metric, []))
            treatment_vals = np.array(self.collector.treatment_metrics.get(metric, []))

            if len(control_vals) > 0 and len(treatment_vals) > 0:
                statistical_results[metric] = {
                    "t_test": self.tester.t_test(control_vals, treatment_vals),
                    "mann_whitney": self.tester.mann_whitney(control_vals, treatment_vals),
                    "bootstrap_ci": self.tester.bootstrap_ci(control_vals, treatment_vals)
                }

        return {
            "test_name": self.config.test_name,
            "status": self.status,
            "duration_days": (datetime.utcnow() - self.start_time).days,
            "config": {
                "control_model": self.config.control_model,
                "treatment_model": self.config.treatment_model,
                "traffic_split": self.config.traffic_split,
                "primary_metric": self.config.primary_metric
            },
            "summary": summary,
            "statistical_tests": statistical_results,
            "recommendation": self._get_recommendation(statistical_results)
        }

    def _get_recommendation(self, results: Dict) -> str:
        """Generate human-readable recommendation."""
        primary = self.config.primary_metric
        if primary not in results:
            return "Insufficient data for recommendation"

        test_result = results[primary]["t_test"]

        if not test_result["significant"]:
            return f"No significant difference detected in {primary}. Continue test or declare equivalence."

        if test_result["relative_lift"] > 0:
            return f"Promote treatment ({self.config.treatment_model}): significant +{test_result['relative_lift']:.2%} lift in {primary}"
        else:
            return f"Keep control ({self.config.control_model}): treatment underperforms by {abs(test_result['relative_lift']):.2%} in {primary}"

    def stop(self):
        """Manually stop the test."""
        self.status = "stopped"
        logger.info("A/B test manually stopped")

    def promote_treatment(self):
        """Promote treatment to 100% traffic."""
        self.status = "promoted"
        self.config.traffic_split = 1.0
        logger.info("Treatment promoted to 100% traffic")

    def rollback(self):
        """Rollback to control (0% treatment traffic)."""
        self.status = "rolled_back"
        self.config.traffic_split = 0.0
        logger.info("Rolled back to control model")
