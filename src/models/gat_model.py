"""
Graph Attention Network (GAT) with multi-head attention for social graph analysis.
Alternative to GraphSAGE with learnable edge weights.
"""
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


@dataclass
class GATConfig:
    in_channels: int = 64
    hidden_channels: int = 128
    out_channels: int = 64
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.2
    edge_dim: Optional[int] = None  # If edge features available
    activation: str = "elu"
    concat: bool = False  # If True, multi-head concat; else average
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GraphAttentionNetwork(nn.Module):
    """
    Production GAT for user-user interaction graphs.
    Supports edge features, residual connections, and layer normalization.
    """
    
    def __init__(self, config: GATConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Input projection if needed
        self.input_proj = nn.Linear(config.in_channels, config.hidden_channels).to(self.device)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.residuals = nn.ModuleList()
        
        for i in range(config.num_layers):
            in_ch = config.hidden_channels if i > 0 else config.hidden_channels
            out_ch = config.hidden_channels if i < config.num_layers - 1 else config.out_channels
            
            # Last layer uses single head for stability
            heads = 1 if i == config.num_layers - 1 else config.num_heads
            concat = False if i == config.num_layers - 1 else config.concat
            
            self.convs.append(
                GATConv(
                    in_channels=in_ch,
                    out_channels=out_ch // heads if concat else out_ch,
                    heads=heads,
                    concat=concat,
                    dropout=config.dropout,
                    edge_dim=config.edge_dim,
                    add_self_loops=True,
                    bias=True
                ).to(self.device)
            )
            
            self.bns.append(BatchNorm(out_ch).to(self.device))
            
            # Residual projection if dimensions change
            if in_ch != out_ch:
                self.residuals.append(nn.Linear(in_ch, out_ch).to(self.device))
            else:
                self.residuals.append(nn.Identity())
        
        self.dropout = nn.Dropout(config.dropout)
        self.activation = getattr(F, config.activation) if hasattr(F, config.activation) else F.elu
        
        logger.info(f"GAT initialized: {config.num_layers} layers, {config.num_heads} heads")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) optional edge features
            batch: (N,) batch assignment for batched graphs
        Returns:
            (N, out_channels) node embeddings
        """
        x = self.input_proj(x)
        
        for i, (conv, bn, res) in enumerate(zip(self.convs, self.bns, self.residuals)):
            x_res = res(x)
            
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
            
            # Residual connection
            x = x + x_res
        
        return x
    
    def predict_node_risk(self, x: torch.Tensor, edge_index: torch.Tensor,
                          node_ids: torch.Tensor) -> torch.Tensor:
        """Predict cancellation risk for specific nodes."""
        embeddings = self.forward(x, edge_index)
        node_emb = embeddings[node_ids]
        # Simple MLP head for risk scoring
        risk_score = torch.sigmoid(node_emb.mean(dim=-1))
        return risk_score
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor,
                              edge_attr: Optional[torch.Tensor] = None,
                              layer: int = 0) -> torch.Tensor:
        """Extract attention weights for interpretability."""
        x = self.input_proj(x)
        for i in range(layer):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = self.activation(x)
        
        # Get attention from target layer
        with torch.no_grad():
            # GATConv stores attention weights after forward
            _ = self.convs[layer](x, edge_index, edge_attr=edge_attr)
            attn = self.convs[layer]._alpha  # (E, heads)
        return attn
    
    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "config": self.config}, path)
        logger.info(f"GAT saved to {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
        logger.info(f"GAT loaded from {path}")
