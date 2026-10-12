"""Cross-modal fusion with attention mechanisms.

Senior-level implementation featuring:
- Multi-head cross-attention between graph and text embeddings
- Gated fusion with learnable modality weights
- Residual connections and layer normalization
- Optional transformer-style fusion layers

When vision embeddings are present they arrive at 512d — the output of the
Qwen2-VL-7B NaViT vision tower (raw 3584d) after the learned projection head.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossModalAttention(nn.Module):
    """Multi-head cross-attention between graph and text modalities.

    Graph embeddings attend to text embeddings and vice versa,
    producing modality-aware fused representations.
    """

    def __init__(
        self,
        graph_dim: int = 128,
        text_dim: int = 4096,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert graph_dim % num_heads == 0, "graph_dim must be divisible by num_heads"
        assert text_dim % num_heads == 0, "text_dim must be divisible by num_heads"

        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = max(graph_dim, text_dim) // num_heads
        self.embed_dim = self.head_dim * num_heads

        # Projection layers to common embedding space
        self.graph_proj = nn.Linear(graph_dim, self.embed_dim)
        self.text_proj = nn.Linear(text_dim, self.embed_dim)

        # Cross-attention: graph queries attend to text keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self-attention for intra-modal refinement
        self.self_attn_graph = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_text = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward networks
        self.ffn_graph = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1_graph = nn.LayerNorm(self.embed_dim)
        self.norm2_graph = nn.LayerNorm(self.embed_dim)
        self.norm1_text = nn.LayerNorm(self.embed_dim)
        self.norm2_text = nn.LayerNorm(self.embed_dim)

        # Output projections back to original dims
        self.graph_out = nn.Linear(self.embed_dim, graph_dim)
        self.text_out = nn.Linear(self.embed_dim, text_dim)

    def forward(
        self,
        graph_emb: torch.Tensor,  # [batch, graph_dim]
        text_emb: torch.Tensor,  # [batch, text_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with cross-modal attention.

        Returns:
            enhanced_graph: [batch, graph_dim]
            enhanced_text: [batch, text_dim]
        """
        batch_size = graph_emb.size(0)

        # Add sequence dimension for attention: [batch, 1, dim]
        g = graph_emb.unsqueeze(1)
        t = text_emb.unsqueeze(1)

        # Project to common space
        g_proj = self.graph_proj(g)  # [batch, 1, embed_dim]
        t_proj = self.text_proj(t)  # [batch, 1, embed_dim]

        # Cross-attention: graph attends to text
        g_cross, _ = self.cross_attn(query=g_proj, key=t_proj, value=t_proj)
        g_cross = self.norm1_graph(g_proj + g_cross)
        g_ffn = self.ffn_graph(g_cross)
        g_enhanced = self.norm2_graph(g_cross + g_ffn)

        # Cross-attention: text attends to graph
        t_cross, _ = self.cross_attn(query=t_proj, key=g_proj, value=g_proj)
        t_cross = self.norm1_text(t_proj + t_cross)
        t_ffn = self.ffn_text(t_cross)
        t_enhanced = self.norm2_text(t_cross + t_ffn)

        # Remove sequence dimension and project back
        enhanced_graph = self.graph_out(g_enhanced.squeeze(1))
        enhanced_text = self.text_out(t_enhanced.squeeze(1))

        # Residual connections to original embeddings
        enhanced_graph = enhanced_graph + graph_emb
        enhanced_text = enhanced_text + text_emb

        return enhanced_graph, enhanced_text


class GatedFusion(nn.Module):
    """Gated multimodal fusion for Graph (128d), Text (4096d), and Vision (512d)."""

    def __init__(
        self,
        graph_dim: int = 128,
        text_dim: int = 4096,
        vision_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim

        # Modality transforms
        self.graph_transform = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vision_transform = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3-way softmax gating mechanism
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        # Joint classification head
        self.joint = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        graph_emb: Optional[torch.Tensor] = None,
        vision_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Backward compatibility: support legacy positional signature forward(graph_emb, text_emb)
        if text_emb is not None and graph_emb is not None:
            if text_emb.shape[-1] == self.graph_dim and graph_emb.shape[-1] == self.text_dim:
                graph_emb, text_emb = text_emb, graph_emb

        primary_tensor = text_emb if text_emb is not None else (graph_emb if graph_emb is not None else vision_emb)
        if primary_tensor is None:
            raise ValueError("At least one modality embedding must be provided to GatedFusion")

        batch_size = primary_tensor.size(0)
        device = primary_tensor.device

        t = self.text_transform(text_emb) if text_emb is not None else torch.zeros(batch_size, self.hidden_dim, device=device)
        g = self.graph_transform(graph_emb) if graph_emb is not None else torch.zeros_like(t)
        v = self.vision_transform(vision_emb) if vision_emb is not None else torch.zeros_like(t)

        gate_in = torch.cat([g, t, v], dim=-1)
        gate_weights = F.softmax(self.gate_network(gate_in), dim=-1)  # [B, 3]

        # Zero out missing modalities in gating
        mask = torch.tensor(
            [graph_emb is not None, text_emb is not None, vision_emb is not None],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        masked_gates = gate_weights * mask
        norm_gates = masked_gates / (masked_gates.sum(dim=-1, keepdim=True) + 1e-8)

        fused = (
            norm_gates[:, 0:1] * g
            + norm_gates[:, 1:2] * t
            + norm_gates[:, 2:3] * v
        )

        self.last_gate = norm_gates.detach()
        return self.joint(fused)


class TransformerFusion(nn.Module):
    """Transformer-based fusion with stacked cross-modal layers.

    Stacks multiple cross-attention + self-attention layers
    for deep cross-modal interaction.
    """

    def __init__(
        self,
        graph_dim: int = 128,
        text_dim: int = 4096,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Project to common dimension
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Stack of cross-modal transformer layers
        self.layers = nn.ModuleList(
            [CrossModalTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, graph_emb, text_emb):
        g = self.graph_proj(graph_emb).unsqueeze(1)  # [B, 1, H]
        t = self.text_proj(text_emb).unsqueeze(1)  # [B, 1, H]

        for layer in self.layers:
            g, t = layer(g, t)

        # Concatenate final representations
        fused = torch.cat([g.squeeze(1), t.squeeze(1)], dim=-1)
        return self.classifier(fused)


class CrossModalTransformerLayer(nn.Module):
    """Single cross-modal transformer layer."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, g, t):
        # Cross-attention
        g_out, _ = self.cross_attn(g, t, t)
        g = self.norm1(g + g_out)

        t_out, _ = self.cross_attn(t, g, g)
        t = self.norm1(t + t_out)

        # Self-attention
        g_out, _ = self.self_attn(g, g, g)
        g = self.norm2(g + g_out)

        t_out, _ = self.self_attn(t, t, t)
        t = self.norm2(t + t_out)

        # FFN
        g = self.norm3(g + self.ffn(g))
        t = self.norm3(t + self.ffn(t))

        return g, t


# Factory for easy switching
FUSION_REGISTRY = {
    "mlp": "src.models.gnn_model.FusionMLP",
    "cross_attention": CrossModalAttention,
    "gated": GatedFusion,
    "transformer": TransformerFusion,
}


def get_fusion_module(fusion_type: str = "gated", **kwargs):
    """Factory function to get fusion module by name."""
    if fusion_type == "mlp":
        from src.models.gnn_model import FusionMLP

        return FusionMLP(**kwargs)
    elif fusion_type == "cross_attention":
        return CrossModalAttention(**kwargs)
    elif fusion_type == "gated":
        return GatedFusion(**kwargs)
    elif fusion_type == "transformer":
        return TransformerFusion(**kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
