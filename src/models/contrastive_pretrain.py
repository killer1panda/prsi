"""
Contrastive pretraining for user embeddings using SimCLR-style NT-Xent loss.
Learns robust user representations from augmented social graph views.
"""
import logging
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveConfig:
    embedding_dim: int = 128
    projection_dim: int = 64
    temperature: float = 0.07
    batch_size: int = 256
    num_epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    augment_dropout: float = 0.1
    augment_mask_ratio: float = 0.15


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR)."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (B, projection_dim) first augmented view
            z_j: (B, projection_dim) second augmented view
        Returns:
            Scalar loss
        """
        B = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        z = F.normalize(z, dim=-1)
        
        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)
        
        # Mask out self-similarities
        mask = torch.eye(2 * B, device=z.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_sim = torch.cat([
            torch.diag(sim, B),
            torch.diag(sim, -B)
        ])  # (2B,)
        
        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        
        # Denominator: sum over all negatives
        denom = torch.exp(sim).sum(dim=1)
        
        # Loss: -log(exp(pos) / sum(exp(all)))
        loss = -pos_sim + torch.log(denom)
        return loss.mean()


class UserEmbeddingProjector(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, embedding_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastivePretrainer:
    """
    SimCLR-style pretraining for user embeddings on graph augmentations.
    """
    
    def __init__(self, encoder: nn.Module, config: Optional[ContrastiveConfig] = None):
        self.config = config or ContrastiveConfig()
        self.device = torch.device(self.config.device)
        self.encoder = encoder.to(self.device)
        self.projector = UserEmbeddingProjector(
            self.config.embedding_dim, 
            self.config.projection_dim
        ).to(self.device)
        self.criterion = NTXentLoss(self.config.temperature)
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.projector.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.num_epochs
        )
        logger.info("ContrastivePretrainer initialized")
    
    def augment(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two augmented views of the graph.
        Augmentations: feature dropout, edge dropout, feature masking.
        """
        # View 1: Feature dropout
        x1 = F.dropout(x, p=self.config.augment_dropout, training=True)
        edge_mask1 = torch.rand(edge_index.size(1), device=edge_index.device) > 0.1
        edge_index1 = edge_index[:, edge_mask1]
        
        # View 2: Feature masking
        x2 = x.clone()
        mask = torch.rand(x.size(), device=x.device) < self.config.augment_mask_ratio
        x2[mask] = 0.0
        edge_mask2 = torch.rand(edge_index.size(1), device=edge_index.device) > 0.15
        edge_index2 = edge_index[:, edge_mask2]
        
        return (x1, edge_index1), (x2, edge_index2)
    
    def train_step(self, x: torch.Tensor, edge_index: torch.Tensor) -> float:
        """Single training step."""
        self.encoder.train()
        self.projector.train()
        self.optimizer.zero_grad()
        
        (x1, ei1), (x2, ei2) = self.augment(x, edge_index)
        
        # Encode both views
        h1 = self.encoder(x1, ei1)
        h2 = self.encoder(x2, ei2)
        
        # Project
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        loss = self.criterion(z1, z2)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.projector.parameters()), 
            max_norm=1.0
        )
        
        self.optimizer.step()
        return loss.item()
    
    def fit(self, data_loader: DataLoader, num_epochs: Optional[int] = None):
        """Full pretraining loop."""
        epochs = num_epochs or self.config.num_epochs
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in data_loader:
                if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                    x = batch.x.to(self.device)
                    edge_index = batch.edge_index.to(self.device)
                else:
                    x, edge_index = batch
                    x = x.to(self.device)
                    edge_index = edge_index.to(self.device)
                
                loss = self.train_step(x, edge_index)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            self.scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Contrastive pretraining complete")
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Extract embeddings without projection head."""
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x, edge_index)
    
    def save(self, path: str):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "projector": self.projector.state_dict(),
            "config": self.config
        }, path)
        logger.info(f"Contrastive pretrainer saved to {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.projector.load_state_dict(checkpoint["projector"])
        logger.info(f"Contrastive pretrainer loaded from {path}")
