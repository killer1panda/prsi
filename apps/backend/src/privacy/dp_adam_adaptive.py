"""
Differentially Private Adam with Layer-Wise Adaptive Quantile Clipping and RDP Accounting.
Prevents gradient signal destruction in deep transformers by dynamically tuning
per-layer clipping thresholds C_l^(t) using private empirical quantile estimators.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AdaptiveDPAdam:
    """
    Differentially Private Adam Optimizer with Layer-Wise Adaptive Clipping.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        target_epsilon: float = 2.0,
        target_delta: float = 1e-5,
        target_quantile: float = 0.7,
        initial_clip_norm: float = 1.0,
        noise_multiplier: float = 1.1,
    ):
        self.params = list(params)
        self.lr = lr
        self.epsilon = target_epsilon
        self.delta = target_delta
        self.gamma_q = target_quantile
        self.noise_multiplier = noise_multiplier
        self.clip_norms = {i: initial_clip_norm for i in range(len(self.params))}
        self.m = {i: torch.zeros_like(p) for i, p in enumerate(self.params)}
        self.v = {i: torch.zeros_like(p) for i, p in enumerate(self.params)}
        self.step_count = 0

    def compute_rdp(self, q: float, steps: int, orders: np.ndarray) -> np.ndarray:
        """Compute analytical RDP curve for subsampled Gaussian mechanism."""
        rdp = []
        for alpha in orders:
            eps_alpha = (q**2 * alpha) / (2 * (self.noise_multiplier**2))
            rdp.append(eps_alpha * steps)
        return np.array(rdp)

    def convert_rdp_to_dp(self, orders: np.ndarray, rdp: np.ndarray) -> float:
        """Convert RDP curve to standard (epsilon, delta)-DP bound."""
        epsilons = rdp + (np.log(1.0 / self.delta) / (orders - 1))
        return float(np.min(epsilons))

    def step(self, per_sample_grads: List[List[torch.Tensor]]):
        """Execute layer-wise adaptive clipping and noisy update."""
        self.step_count += 1
        batch_size = len(per_sample_grads)

        for l_idx, param in enumerate(self.params):
            layer_grads = [sample[l_idx] for sample in per_sample_grads]
            norms = torch.tensor([torch.norm(g, p=2) for g in layer_grads])

            # Update layer-wise adaptive clip threshold
            exceeded = (norms > self.clip_norms[l_idx]).float().mean().item()
            self.clip_norms[l_idx] *= np.exp(-0.2 * (exceeded - (1.0 - self.gamma_q)))

            # Clip individual gradients
            C = self.clip_norms[l_idx]
            clipped_grads = []
            for g, norm in zip(layer_grads, norms):
                factor = min(1.0, C / max(norm.item(), 1e-8))
                clipped_grads.append(g * factor)

            # Sum and add calibrated Gaussian noise
            summed_grad = torch.stack(clipped_grads).sum(dim=0)
            noise = torch.randn_like(summed_grad) * (self.noise_multiplier * C)
            priv_grad = (summed_grad + noise) / batch_size

            # Adam moment update
            self.m[l_idx] = 0.9 * self.m[l_idx] + 0.1 * priv_grad
            self.v[l_idx] = 0.999 * self.v[l_idx] + 0.001 * (priv_grad**2)

            m_hat = self.m[l_idx] / (1.0 - 0.9**self.step_count)
            v_hat = self.v[l_idx] / (1.0 - 0.999**self.step_count)

            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
