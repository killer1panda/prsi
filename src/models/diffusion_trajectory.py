"""
Continuous Score-Based 1D Diffusion Model for Social Outrage Cascade Forecasting.
Implements a 1D Diffusion Transformer (DiT-1D) modeling 72-hour stochastic cascade trajectories
conditioned on multimodal context vectors via reverse-time SDE sampling.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiTBlock1D(nn.Module):
    """1D Diffusion Transformer block with adaptive LayerNorm conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, cond_dim: int = 128):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [Batch, SeqLen, HiddenDim]
        # cond: [Batch, CondDim]
        gamma_beta = self.cond_proj(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # Modulated self-attention
        norm_x = self.norm1(x) * (1 + gamma) + beta
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out

        # Modulated MLP
        norm_x2 = self.norm2(x) * (1 + gamma) + beta
        x = x + self.mlp(norm_x2)
        return x


class DiffusionTrajectoryForecaster(nn.Module):
    """
    Score-based 1D Diffusion model for generating hourly outrage cascade curves
    over horizon H = 72 hours conditioned on post and author graph context.
    """

    def __init__(
        self,
        horizon: int = 72,
        hidden_dim: int = 128,
        cond_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_diffusion_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_diffusion_timesteps = num_diffusion_timesteps

        # SDE Noise Schedule (Variance-Preserving)
        betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Encoders
        self.trajectory_proj = nn.Linear(1, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, horizon, hidden_dim) * 0.02)
        self.time_emb = SinusoidalTimeEmbedding(cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # DiT Stack
        self.blocks = nn.ModuleList([
            DiTBlock1D(hidden_dim, num_heads=num_heads, cond_dim=cond_dim)
            for _ in range(num_layers)
        ])

        # Output score head
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        y_noisy: torch.Tensor,
        t: torch.Tensor,
        context_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        y_noisy: [Batch, 72] or [Batch, 72, 1]
        t: [Batch] integer timesteps in [0, num_diffusion_timesteps-1]
        context_embedding: [Batch, cond_dim] multimodal context
        Returns predicted noise epsilon_theta: [Batch, 72]
        """
        if y_noisy.ndim == 2:
            y_noisy = y_noisy.unsqueeze(-1)

        b, seq_len, _ = y_noisy.shape
        x = self.trajectory_proj(y_noisy) + self.pos_emb[:, :seq_len, :]

        # Combine diffusion step embedding and context embedding
        t_emb = self.time_emb(t.float())
        joint_cond = self.cond_mlp(torch.cat([t_emb, context_embedding], dim=-1))

        # Forward through DiT blocks
        for block in self.blocks:
            x = block(x, joint_cond)

        out = self.out_head(self.final_norm(x)).squeeze(-1)
        return out

    def compute_loss(
        self,
        y_0: torch.Tensor,
        context_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Standard DDPM epsilon-prediction MSE loss."""
        device = y_0.device
        batch_size = y_0.shape[0]
        t = torch.randint(0, self.num_diffusion_timesteps, (batch_size,), device=device).long()

        noise = torch.randn_like(y_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(batch_size, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1)

        y_noisy = sqrt_alpha * y_0 + sqrt_one_minus_alpha * noise
        noise_pred = self.forward(y_noisy, t, context_embedding)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample_trajectories(
        self,
        context_embedding: torch.Tensor,
        num_samples: int = 100,
        device: str = "cpu"
    ) -> Dict[str, np.ndarray]:
        """
        Sample K continuous 72-hour outrage cascade curves via reverse DDPM sampling.
        """
        self.eval()
        batch_size = context_embedding.shape[0]
        # Expand context for num_samples
        ctx = context_embedding.repeat_interleave(num_samples, dim=0).to(device)
        total_samples = batch_size * num_samples

        # Initialize from Gaussian noise
        y = torch.randn((total_samples, self.horizon), device=device)

        for step in reversed(range(self.num_diffusion_timesteps)):
            t = torch.full((total_samples,), step, device=device, dtype=torch.long)
            noise_pred = self.forward(y, t, ctx)

            alpha = self.alphas[step]
            alpha_cumprod = self.alphas_cumprod[step]
            beta = self.betas[step]

            if step > 0:
                z = torch.randn_like(y)
                sigma = torch.sqrt(beta * (1.0 - self.alphas_cumprod[step - 1]) / (1.0 - alpha_cumprod))
            else:
                z = torch.zeros_like(y)
                sigma = 0.0

            # Reverse SDE update formula
            y = (1.0 / torch.sqrt(alpha)) * (y - (beta / torch.sqrt(1.0 - alpha_cumprod)) * noise_pred) + sigma * z

        # Reshape to [Batch, NumSamples, Horizon]
        trajectories = y.view(batch_size, num_samples, self.horizon).cpu().numpy()
        # Scale to 0 - 100 doom index bounds
        trajectories = np.clip(trajectories * 25.0 + 50.0, 0.0, 100.0)

        # Compute empirical trajectory bands
        p05 = np.percentile(trajectories, 5, axis=1)
        p50 = np.percentile(trajectories, 50, axis=1)
        p95 = np.percentile(trajectories, 95, axis=1)
        prob_critical = np.mean(np.max(trajectories, axis=2) >= 80.0, axis=1)

        return {
            "all_trajectories": trajectories,
            "median_trajectory": p50,
            "lower_band_p05": p05,
            "upper_band_p95": p95,
            "prob_critical_cascade": prob_critical
        }
