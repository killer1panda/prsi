"""
Temporal Graph Network (TGN) implementation for evolving social graphs.
Models how user interactions change over time to predict future cancellation cascades.
"""
import logging
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass
class TGNConfig:
    node_dim: int = 128
    edge_dim: int = 64
    time_dim: int = 32
    memory_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    message_agg: str = "last"  # last, mean, attention


class TimeEncoder(nn.Module):
    """Encode time differences using sinusoidal positional encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.linspace(0, 1, dim).unsqueeze(0) * 1000)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (N,) time differences in seconds
        Returns:
            (N, dim) time encodings
        """
        t = t.unsqueeze(-1)  # (N, 1)
        phase = t * self.w
        return torch.cos(phase)


class MemoryModule(nn.Module):
    """Memory bank for tracking node states over time."""

    def __init__(self, num_nodes: int, memory_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.memory = nn.Parameter(torch.zeros(num_nodes, memory_dim))
        self.last_update = nn.Parameter(torch.zeros(num_nodes))

        self.msg_func = nn.Sequential(
            nn.Linear(memory_dim * 2 + 64, memory_dim),  # src_mem + dst_mem + edge_feat
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim)
        )
        self.gru = nn.GRUCell(memory_dim, memory_dim)

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.memory[node_ids]

    def update_memory(self, node_ids: torch.Tensor, messages: torch.Tensor):
        """Update memory using GRU."""
        current_mem = self.memory[node_ids]
        updated = self.gru(messages, current_mem)
        self.memory[node_ids] = updated

    def compute_messages(self, src_ids: torch.Tensor, dst_ids: torch.Tensor,
                         edge_feats: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute messages for edge updates."""
        src_mem = self.memory[src_ids]
        dst_mem = self.memory[dst_ids]
        combined = torch.cat([src_mem, dst_mem, edge_feats], dim=-1)
        return self.msg_func(combined)

    def reset(self):
        self.memory.data.zero_()
        self.last_update.data.zero_()


class TemporalGraphNetwork(nn.Module):
    """
    TGN: Temporal Graph Network for dynamic graphs.
    Adapted from Rossi et al., 2020.
    """

    def __init__(self, config: TGNConfig, num_nodes: int):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.num_nodes = num_nodes

        self.time_encoder = TimeEncoder(config.time_dim).to(self.device)
        self.memory = MemoryModule(num_nodes, config.memory_dim).to(self.device)

        # Embedding layers
        self.node_embedding = nn.Embedding(num_nodes, config.node_dim).to(self.device)

        # Graph attention layers for temporal neighbors
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.node_dim + config.time_dim + config.memory_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True
            ).to(self.device)
            for _ in range(config.num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.node_dim + config.time_dim + config.memory_dim).to(self.device)
            for _ in range(config.num_layers)
        ])

        # Output projection
        total_dim = config.node_dim + config.time_dim + config.memory_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(total_dim // 2, config.node_dim)
        ).to(self.device)

        self.dropout = nn.Dropout(config.dropout)
        logger.info(f"TGN initialized: {num_nodes} nodes, {config.num_layers} layers")

    def forward(self, node_ids: torch.Tensor, 
                neighbor_ids: List[torch.Tensor],
                edge_feats: List[torch.Tensor],
                timestamps: List[torch.Tensor],
                time_diffs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with temporal neighborhood sampling.

        Args:
            node_ids: (B,) central node IDs
            neighbor_ids: List of (B, N_i) neighbor IDs per layer
            edge_feats: List of (B, N_i, edge_dim) edge features
            timestamps: List of (B, N_i) neighbor timestamps
            time_diffs: List of (B, N_i) time differences from current time
        Returns:
            (B, node_dim) node embeddings
        """
        B = node_ids.size(0)

        # Get node embeddings and memory
        node_emb = self.node_embedding(node_ids)  # (B, node_dim)
        node_mem = self.memory.get_memory(node_ids)  # (B, memory_dim)

        # Time encoding for current nodes (t=0)
        current_time_enc = self.time_encoder(torch.zeros(B, device=self.device))

        x = torch.cat([node_emb, current_time_enc, node_mem], dim=-1).unsqueeze(1)  # (B, 1, D)

        for layer_idx in range(self.config.num_layers):
            if layer_idx < len(neighbor_ids):
                nbr_ids = neighbor_ids[layer_idx]  # (B, N)
                nbr_emb = self.node_embedding(nbr_ids)  # (B, N, node_dim)
                nbr_mem = self.memory.get_memory(nbr_ids.view(-1)).view(B, -1, self.config.memory_dim)

                # Time encoding for neighbors
                td = time_diffs[layer_idx]  # (B, N)
                td_flat = td.view(-1)
                time_enc = self.time_encoder(td_flat).view(B, -1, self.config.time_dim)

                # Combine neighbor features
                nbr_combined = torch.cat([nbr_emb, time_enc, nbr_mem], dim=-1)  # (B, N, D)

                # Self-attention between central node and neighbors
                attn_out, _ = self.attention_layers[layer_idx](
                    x, nbr_combined, nbr_combined
                )
                x = x + self.dropout(attn_out)
                x = self.layer_norms[layer_idx](x)

        x = x.squeeze(1)  # (B, D)
        return self.output_mlp(x)

    def update_memory_from_batch(self, src_ids: torch.Tensor, dst_ids: torch.Tensor,
                                  edge_feats: torch.Tensor, timestamps: torch.Tensor):
        """Update memory after processing a batch of edges."""
        messages = self.memory.compute_messages(src_ids, dst_ids, edge_feats, timestamps)

        # Aggregate messages per node
        unique_nodes = torch.cat([src_ids, dst_ids]).unique()
        for node in unique_nodes:
            mask = (src_ids == node) | (dst_ids == node)
            if mask.any():
                node_messages = messages[mask]
                if self.config.message_agg == "mean":
                    agg_msg = node_messages.mean(dim=0)
                else:
                    agg_msg = node_messages[-1]  # last
                self.memory.update_memory(node.unsqueeze(0), agg_msg.unsqueeze(0))
                self.memory.last_update[node] = timestamps[mask].max()

    def predict_link(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Predict probability of edge between src and dst."""
        src_emb = self.node_embedding(src)
        dst_emb = self.node_embedding(dst)
        score = torch.sum(src_emb * dst_emb, dim=-1)
        return torch.sigmoid(score)

    def reset_memory(self):
        self.memory.reset()
