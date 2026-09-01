"""
Hypergraph Neural Network (HGNN) & Continuous-Time Dynamic Graph Attention (CTDGA).
Models multi-user discussion hyperedges, multi-relational composition (CompGCN),
and continuous temporal event point processes via learnable Hawkes dynamics.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HypergraphConv(nn.Module):
    """
    Hypergraph Convolutional Layer (Feng et al., AAAI 2019).
    X^{(l+1)} = σ(D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X^{(l)} Θ)
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.dropout = dropout
        self.norm = nn.BatchNorm1d(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        hyperedge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: Node feature matrix [N, in_channels]
        hyperedge_index: [2, num_incidence_pairs] where row 0 is node_idx, row 1 is hyperedge_idx
        hyperedge_weight: [E] controversy weights for each hyperedge
        """
        num_nodes = x.size(0)
        num_edges = int(hyperedge_index[1].max().item() + 1) if hyperedge_index.numel() > 0 else 1

        if hyperedge_weight is None:
            hyperedge_weight = torch.ones(num_edges, device=x.device, dtype=x.dtype)

        node_idx = hyperedge_index[0]
        edge_idx = hyperedge_index[1]

        # 1. Degree of hyperedges: D_e = sum of incidence
        ones_nodes = torch.ones_like(node_idx, dtype=x.dtype)
        d_e = torch.zeros(num_edges, device=x.device, dtype=x.dtype)
        d_e.scatter_add_(0, edge_idx, ones_nodes)
        d_e = torch.clamp(d_e, min=1.0)
        inv_d_e = 1.0 / d_e

        # 2. Degree of vertices: D_v = sum of weighted incidence
        weights_per_pair = hyperedge_weight[edge_idx]
        d_v = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        d_v.scatter_add_(0, node_idx, weights_per_pair)
        d_v = torch.clamp(d_v, min=1.0)
        inv_sqrt_d_v = torch.rsqrt(d_v)

        # 3. Node to Hyperedge Message Passing: H^T D_v^{-1/2} X
        x_norm = x * inv_sqrt_d_v.unsqueeze(-1)
        node_feats_per_pair = x_norm[node_idx]

        # Aggregate to hyperedges
        edge_repr = torch.zeros(num_edges, self.in_channels, device=x.device, dtype=x.dtype)
        edge_repr.index_add_(0, edge_idx, node_feats_per_pair)

        # Multiply by W_e D_e^{-1}
        edge_repr = edge_repr * (hyperedge_weight * inv_d_e).unsqueeze(-1)

        # 4. Hyperedge to Node Message Passing: H (W_e D_e^{-1} H^T D_v^{-1/2} X)
        edge_feats_per_pair = edge_repr[edge_idx]
        out_nodes = torch.zeros(num_nodes, self.in_channels, device=x.device, dtype=x.dtype)
        out_nodes.index_add_(0, node_idx, edge_feats_per_pair)

        # Multiply by D_v^{-1/2} and linear projection Theta
        out = (out_nodes * inv_sqrt_d_v.unsqueeze(-1)) @ self.weight + self.bias
        out = F.relu(self.norm(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class ContinuousTimeHawkesGAT(nn.Module):
    """
    Continuous-Time Dynamic Graph Attention with learnable Hawkes Point Process intensity:
    λ(t | H_t) = μ + Σ α_k exp(-β (t - t_k)) * Softmax(q_u(t)^T k_v(t_k) / sqrt(d))
    """

    def __init__(self, node_dim: int = 128, time_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.node_dim = node_dim
        self.time_dim = time_dim
        self.num_heads = num_heads

        # Sinusoidal continuous-time basis weights
        self.time_w = nn.Parameter(torch.randn(time_dim) * 0.1)

        # Hawkes self-excitation decay rate β and baseline intensity μ
        self.beta = nn.Parameter(torch.tensor([0.1]))
        self.mu = nn.Parameter(torch.tensor([1.0]))
        self.alpha_excitation = nn.Parameter(torch.tensor([0.5]))

        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim + time_dim, node_dim)
        self.v_proj = nn.Linear(node_dim + time_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)

    def encode_time(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """Continuous time delta mapping: Δt -> cos(Δt * w)."""
        time_deltas = time_deltas.unsqueeze(-1)
        return torch.cos(time_deltas * self.time_w.unsqueeze(0))

    def forward(
        self, target_node_emb: torch.Tensor, neighbor_embs: torch.Tensor, time_deltas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        target_node_emb: [Batch, NodeDim]
        neighbor_embs: [Batch, NumNeighbors, NodeDim]
        time_deltas: [Batch, NumNeighbors] in hours/seconds
        Returns: (aggregated_embedding [Batch, NodeDim], hawkes_intensity [Batch])
        """
        t_enc = self.encode_time(time_deltas)
        k_in = torch.cat([neighbor_embs, t_enc], dim=-1)

        q = self.q_proj(target_node_emb).unsqueeze(1)  # [B, 1, D]
        k = self.k_proj(k_in)  # [B, K, D]
        v = self.v_proj(k_in)  # [B, K, D]

        # Scaled dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.node_dim)
        attn_weights = F.softmax(scores, dim=-1)

        # Hawkes temporal excitation weight: exp(-beta * delta_t)
        decay = torch.exp(-F.softplus(self.beta) * torch.clamp(time_deltas, min=0.0)).unsqueeze(1)
        modulated_weights = attn_weights * decay
        modulated_weights = modulated_weights / torch.clamp(
            modulated_weights.sum(dim=-1, keepdim=True), min=1e-6
        )

        out = torch.bmm(modulated_weights, v).squeeze(1)
        out = self.out_proj(out)

        # Compute instant point process intensity
        instant_intensity = F.softplus(self.mu) + F.softplus(self.alpha_excitation) * decay.squeeze(
            1
        ).sum(dim=-1)

        return out, instant_intensity


class FrontierHypergraphGNN(nn.Module):
    """
    Combined Frontier Hypergraph & Relational Architecture.
    Ingests node attributes, conversation hyperedges, and temporal interaction timestamps.
    """

    def __init__(self, in_features: int = 6, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.hg_conv1 = HypergraphConv(hidden_dim, hidden_dim)
        self.hg_conv2 = HypergraphConv(hidden_dim, hidden_dim)
        self.hawkes_gat = ContinuousTimeHawkesGAT(hidden_dim, time_dim=32)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        hyperedge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = F.gelu(self.input_proj(x))
        h = self.hg_conv1(h, hyperedge_index, hyperedge_weight)
        h = self.hg_conv2(h, hyperedge_index, hyperedge_weight)
        return self.out_proj(h)


class HypergraphHGNN(nn.Module):
    """
    Hypergraph Neural Network (HGNN) using incidence matrix convolution.
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphConv(in_channels, hidden_channels, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(HypergraphConv(hidden_channels, hidden_channels, dropout=dropout))
        self.convs.append(HypergraphConv(hidden_channels, out_channels, dropout=dropout))

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, hyperedge_index, hyperedge_weight))
        x = self.convs[-1](x, hyperedge_index, hyperedge_weight)
        return x
