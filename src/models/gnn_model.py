"""GraphSAGE + Multimodal Fusion Model.

GraphSAGE for user network embeddings + DistilBERT for text embeddings,
fused via MLP for final cancellation prediction.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import DistilBertModel, DistilBertTokenizer

logger = logging.getLogger(__name__)


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder for user network embeddings."""

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
        self.batch_norms = nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer (no activation, no dropout)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        x = self.batch_norms[-1](x)

        return x

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Get node embeddings without final activation."""
        return self.forward(x, edge_index, edge_weight)


class TextEncoder(nn.Module):
    """DistilBERT-based text encoder."""

    def __init__(self, model_name: str = "distilbert-base-uncased", freeze_layers: int = 5):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # 768

        # Freeze bottom layers for faster training / less overfitting
        if freeze_layers > 0:
            for param in self.bert.parameters():
                param.requires_grad = False
            # Unfreeze top layers
            for layer in self.bert.transformer.layer[freeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(f"Frozen DistilBERT layers 0-{freeze_layers-1}, unfrozen {freeze_layers}-5")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def encode_text(self, text: str, tokenizer, device="cuda"):
        """Encode a single text string."""
        self.eval()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            embedding = self.forward(inputs['input_ids'], inputs['attention_mask'])

        return embedding.squeeze(0)


class FusionMLP(nn.Module):
    """Fusion layer combining graph and text embeddings."""

    def __init__(
        self,
        graph_dim: int = 128,
        text_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()

        total_dim = graph_dim + text_dim

        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_emb, text_emb):
        # Concatenate
        x = torch.cat([graph_emb, text_emb], dim=-1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x


class MultimodalDoomPredictor(nn.Module):
    """End-to-end multimodal doom predictor.

    Combines GraphSAGE (user network) + DistilBERT (text) + Fusion MLP.
    """

    def __init__(
        self,
        graph_in_channels: int = 6,
        graph_hidden: int = 128,
        graph_out: int = 128,
        graph_layers: int = 2,
        text_model: str = "distilbert-base-uncased",
        text_freeze: int = 5,
        fusion_hidden: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.graph_encoder = GraphSAGEEncoder(
            in_channels=graph_in_channels,
            hidden_channels=graph_hidden,
            out_channels=graph_out,
            num_layers=graph_layers,
            dropout=dropout,
        )

        self.text_encoder = TextEncoder(
            model_name=text_model,
            freeze_layers=text_freeze,
        )

        self.fusion = FusionMLP(
            graph_dim=graph_out,
            text_dim=768,
            hidden_dim=fusion_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained(text_model)

    def forward(
        self,
        x,
        edge_index,
        input_ids,
        attention_mask,
        user_indices,
        edge_weight=None,
    ):
        """Forward pass for batch of (user, text) pairs.

        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Graph edges [2, num_edges]
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            user_indices: Index of user node for each sample [batch_size]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            logits: [batch_size, num_classes]
        """
        # Graph embeddings for all nodes
        graph_embeddings = self.graph_encoder(x, edge_index, edge_weight)  # [num_nodes, graph_out]

        # Select user embeddings for batch
        user_embeddings = graph_embeddings[user_indices]  # [batch_size, graph_out]

        # Text embeddings
        text_embeddings = self.text_encoder(input_ids, attention_mask)  # [batch_size, 768]

        # Fusion
        logits = self.fusion(user_embeddings, text_embeddings)  # [batch_size, num_classes]

        return logits

    def predict(
        self,
        x,
        edge_index,
        text: str,
        user_idx: int,
        edge_weight=None,
        device="cuda",
    ) -> Tuple[int, float]:
        """Predict for a single (user, text) pair.

        Returns:
            prediction: 0 or 1
            probability: float in [0, 1]
        """
        self.eval()

        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        # Move graph to device
        x = x.to(device)
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        with torch.no_grad():
            logits = self.forward(
                x, edge_index,
                inputs['input_ids'], inputs['attention_mask'],
                torch.tensor([user_idx], dtype=torch.long, device=device),
                edge_weight,
            )
            probs = F.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1).item()
            prob = probs[0, 1].item()  # Probability of class 1 (doom)

        return pred, prob

    def get_multimodal_embeddings(
        self,
        x,
        edge_index,
        text: str,
        user_idx: int,
        edge_weight=None,
        device="cuda",
    ) -> dict:
        """Get intermediate embeddings for interpretability."""
        self.eval()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        x = x.to(device)
        edge_index = edge_index.to(device)

        with torch.no_grad():
            graph_emb = self.graph_encoder(x, edge_index, edge_weight)
            user_emb = graph_emb[user_idx]
            text_emb = self.text_encoder(inputs['input_ids'], inputs['attention_mask'])

        return {
            'graph_embedding': user_emb.cpu().numpy(),
            'text_embedding': text_emb.squeeze(0).cpu().numpy(),
            'combined_dim': user_emb.shape[-1] + text_emb.shape[-1],
        }


if __name__ == "__main__":
    # Quick sanity check
    model = MultimodalDoomPredictor()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Graph encoder:", model.graph_encoder)
    print("Fusion MLP:", model.fusion)
