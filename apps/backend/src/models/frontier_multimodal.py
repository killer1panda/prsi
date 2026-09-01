"""
Frontier Vision-Language Architecture (Qwen2-VL-7B + Mistral-7B QLoRA + Q-Former).
Implements a Multimodal Perceiver Resampler (Q-Former) with learnable latent query tokens
and InfoNCE cross-modal contrastive alignment for nuanced meme & text outrage modeling.

Vision backbone: Qwen2-VL-7B NaViT with native dynamic resolution (up to 1120×1120).
Raw vision hidden size: 3584d. Passes through Q-Former before language fusion.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    Quantized / Low-Rank Adaptation (LoRA) layer:
    W = W_0 + (alpha / r) * B @ A
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r

        # Frozen base weight
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # Trainable low-rank adapters
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out


class MultimodalQFormer(nn.Module):
    """
    Multimodal Perceiver Resampler / Q-Former Bridge.
    Uses learnable query tokens to compress arbitrary vision patch sequences
    (from the Qwen2-VL-7B NaViT tower, hidden_size=3584) into fixed-length
    latent outrage representations for text fusion.
    """

    def __init__(
        self,
        num_queries: int = 32,
        query_dim: int = 4096,
        vision_dim: int = 3584,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, query_dim) * 0.02)

        # Cross-Attention: Query attends to Vision Patches
        self.vision_proj = nn.Linear(vision_dim, query_dim)
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_vision = nn.LayerNorm(query_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=num_heads, batch_first=True
        )

        # Self-Attention & FFN
        self.self_attn = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=num_heads, batch_first=True
        )
        self.norm_self = nn.LayerNorm(query_dim)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4), nn.GELU(), nn.Linear(query_dim * 4, query_dim)
        )
        self.norm_mlp = nn.LayerNorm(query_dim)

    def forward(self, vision_patches: torch.Tensor) -> torch.Tensor:
        """
        vision_patches: [Batch, NumPatches, VisionDim]
        Returns: [Batch, NumQueries, QueryDim] compressed latent vision representation
        """
        batch_size = vision_patches.size(0)
        q = self.queries.expand(batch_size, -1, -1)
        v = self.vision_proj(vision_patches)

        # 1. Cross-Attention
        norm_q = self.norm_query(q)
        norm_v = self.norm_vision(v)
        attn_out, _ = self.cross_attn(query=norm_q, key=norm_v, value=norm_v)
        q = q + attn_out

        # 2. Self-Attention
        norm_q2 = self.norm_self(q)
        self_out, _ = self.self_attn(query=norm_q2, key=norm_q2, value=norm_q2)
        q = q + self_out

        # 3. Feedforward
        q = q + self.mlp(self.norm_mlp(q))
        return q


class FrontierMultimodalPredictor(nn.Module):
    """
    Unified Frontier Multimodal Architecture.
    Combines Vision Patches (Qwen2-VL-7B, 3584d hidden → Q-Former compressed),
    Text Sequences (Mistral QLoRA), and Graph Latents via Q-Former cross-attention
    and InfoNCE alignment. Vision patches are fed raw (3584d) into the Q-Former;
    after compression they are projected to latent_dim for joint transformer fusion.
    """

    def __init__(
        self,
        vision_dim: int = 3584,  # Qwen2-VL-7B NaViT hidden size (before Q-Former)
        text_dim: int = 1024,
        graph_dim: int = 128,
        latent_dim: int = 512,
        num_classes: int = 2,
    ):
        super().__init__()
        self.q_former = MultimodalQFormer(
            num_queries=16, query_dim=latent_dim, vision_dim=vision_dim
        )
        self.text_proj = LoRALinear(text_dim, latent_dim, r=16)
        self.graph_proj = nn.Linear(graph_dim, latent_dim)

        # Joint Multimodal Attention
        self.joint_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=8, dim_feedforward=latent_dim * 4, batch_first=True
            ),
            num_layers=2,
        )

        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        vision_patches: Optional[torch.Tensor],
        text_tokens: torch.Tensor,
        graph_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        vision_patches: [B, N_v, vision_dim] or None
        text_tokens: [B, N_t, text_dim]
        graph_emb: [B, graph_dim] or None
        """
        batch_size = text_tokens.size(0)
        tokens_list = []

        # 1. Project Text
        text_latents = self.text_proj(text_tokens)
        tokens_list.append(text_latents)

        # 2. Compress Vision if present
        if vision_patches is not None:
            vision_latents = self.q_former(vision_patches)
            tokens_list.append(vision_latents)

        # 3. Add Graph Context if present
        if graph_emb is not None:
            graph_latents = self.graph_proj(graph_emb).unsqueeze(1)
            tokens_list.append(graph_latents)

        # Concatenate tokens along sequence dimension
        joint_seq = torch.cat(tokens_list, dim=1)
        fused_seq = self.joint_transformer(joint_seq)

        # Global mean pooling for classification
        pooled = fused_seq.mean(dim=1)
        logits = self.head(pooled)

        return {"logits": logits, "pooled_embedding": pooled, "fused_sequence": fused_seq}

    def compute_infonce_loss(
        self, z_vision: torch.Tensor, z_text: torch.Tensor, temperature: float = 0.07
    ) -> torch.Tensor:
        """Symmetric InfoNCE cross-modal alignment loss."""
        z_v_norm = F.normalize(z_vision, p=2, dim=-1)
        z_t_norm = F.normalize(z_text, p=2, dim=-1)

        sim_matrix = (z_v_norm @ z_t_norm.t()) / temperature
        labels = torch.arange(z_vision.size(0), device=z_vision.device)

        loss_v2t = F.cross_entropy(sim_matrix, labels)
        loss_t2v = F.cross_entropy(sim_matrix.t(), labels)
        return (loss_v2t + loss_t2v) / 2.0
