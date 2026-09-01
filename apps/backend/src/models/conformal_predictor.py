"""
Finite-Sample Distribution-Free Split Conformal Prediction for Social Outrage Forecasting.
Implements Conformalized Quantile Regression (CQR) with mathematical coverage guarantees:
P(y ∈ [L, U]) >= 1 - α (Romano, Patterson, Candès, NeurIPS 2019).
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QuantileLoss(nn.Module):
    """Pinball / Quantile loss for training quantile regression models."""

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds: [Batch, num_quantiles]
        target: [Batch] or [Batch, 1]
        """
        if target.ndim == 1:
            target = target.unsqueeze(-1)

        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i : i + 1]
            loss = torch.max((q - 1) * error, q * error)
            losses.append(loss.mean())
        return sum(losses) / len(losses)


class QuantileNeuralNetwork(nn.Module):
    """Multi-head quantile MLP for predicting lower, median, and upper quantiles."""

    def __init__(
        self,
        in_features: int = 128,
        hidden_dim: int = 256,
        quantiles: List[float] = [0.05, 0.50, 0.95],
    ):
        super().__init__()
        self.quantiles = quantiles
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        # Predict one output per quantile
        self.heads = nn.Linear(hidden_dim // 2, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        out = self.heads(h)
        # Ensure monotonic quantile sorting: q_low <= q_med <= q_high
        q_sorted, _ = torch.sort(out, dim=-1)
        return q_sorted


class ConformalPredictor:
    """
    Split Conformalized Quantile Regressor (CQR).
    Computes distribution-free non-conformity scores and conformal multipliers
    providing rigorous finite-sample coverage guarantees.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        reach_thresholds: Dict[str, Tuple[int, int]] = {
            "nano": (0, 1000),
            "micro": (1000, 50000),
            "macro": (50000, 500000),
            "mega": (500000, int(1e9)),
        },
    ):
        self.alpha = alpha
        self.reach_thresholds = reach_thresholds
        self.global_conformal_multiplier: float = 0.0
        self.stratified_multipliers: Dict[str, float] = {}
        self.is_calibrated: bool = False

    def _get_reach_tier(self, followers: int) -> str:
        for tier, (low, high) in self.reach_thresholds.items():
            if low <= followers < high:
                return tier
        return "mega"

    def calibrate(
        self,
        y_true: np.ndarray,
        q_low_preds: np.ndarray,
        q_high_preds: np.ndarray,
        followers_array: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calibrate on held-out calibration dataset D_cal = {(x_i, y_i)}_{i=1}^n.
        Computes non-conformity score E_i = max(q_low - y_i, y_i - q_high).
        """
        n = len(y_true)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Non-conformity scores
        e_scores = np.maximum(q_low_preds - y_true, y_true - q_high_preds)

        # Finite-sample (1 - alpha) conformal multiplier
        q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))
        self.global_conformal_multiplier = float(np.quantile(e_scores, q_level, method="higher"))

        # Stratified calibration by reach tiers
        if followers_array is not None and len(followers_array) == n:
            tier_func = np.vectorize(self._get_reach_tier)
            tiers = tier_func(followers_array)

            for tier in self.reach_thresholds.keys():
                mask = tiers == tier
                if np.sum(mask) >= 15:
                    tier_scores = e_scores[mask]
                    n_tier = len(tier_scores)
                    tier_q_level = min(
                        1.0, max(0.0, np.ceil((n_tier + 1) * (1.0 - self.alpha)) / n_tier)
                    )
                    self.stratified_multipliers[tier] = float(
                        np.quantile(tier_scores, tier_q_level, method="higher")
                    )
                else:
                    self.stratified_multipliers[tier] = self.global_conformal_multiplier

        self.is_calibrated = True
        return {
            "global_multiplier": self.global_conformal_multiplier,
            "stratified_multipliers": self.stratified_multipliers,
            "target_coverage": 1.0 - self.alpha,
        }

    def predict_intervals(
        self,
        q_low_preds: np.ndarray,
        q_high_preds: np.ndarray,
        q_med_preds: Optional[np.ndarray] = None,
        followers_array: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate conformalized prediction intervals:
        Lower Bound = q_low - Q_{1-α}
        Upper Bound = q_high + Q_{1-α}
        """
        if not self.is_calibrated:
            logger.warning("ConformalPredictor not calibrated. Using default 0 multiplier.")
            multipliers = np.zeros_like(q_low_preds)
        elif followers_array is not None and self.stratified_multipliers:
            multipliers = np.array(
                [
                    self.stratified_multipliers.get(
                        self._get_reach_tier(f), self.global_conformal_multiplier
                    )
                    for f in followers_array
                ]
            )
        else:
            multipliers = np.full_like(q_low_preds, self.global_conformal_multiplier)

        lower_bounds = np.clip(q_low_preds - multipliers, 0.0, 100.0)
        upper_bounds = np.clip(q_high_preds + multipliers, 0.0, 100.0)

        # Ensure lower <= upper
        lower_bounds = np.minimum(lower_bounds, upper_bounds)

        results = {
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds,
            "interval_length": upper_bounds - lower_bounds,
        }
        if q_med_preds is not None:
            results["point_prediction"] = np.clip(q_med_preds, lower_bounds, upper_bounds)

        return results

    def evaluate_coverage(
        self, y_test: np.ndarray, intervals: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate empirical coverage and sharpness."""
        lower = intervals["lower_bound"]
        upper = intervals["upper_bound"]
        covered = (y_test >= lower) & (y_test <= upper)
        empirical_coverage = float(np.mean(covered))
        mean_interval_length = float(np.mean(intervals["interval_length"]))

        return {
            "empirical_coverage": empirical_coverage,
            "target_coverage": 1.0 - self.alpha,
            "coverage_gap": empirical_coverage - (1.0 - self.alpha),
            "mean_interval_length": mean_interval_length,
            "valid_guarantee": empirical_coverage >= (1.0 - self.alpha),
        }
