"""Graph Neural Network + Multimodal Fusion Model.

Hypergraph HGNN + CompGCN + CTDGA (Hawkes Process) for user network embeddings
+ Mistral-7B-Instruct with 4-bit QLoRA for text embeddings, fused via MLP for final cancellation prediction.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

from src.models.hypergraph_gnn import HypergraphHGNN
from src.models.temporal_gnn import CTDGAHawkesEncoder

logger = logging.getLogger(__name__)


class CompGCNEncoder(nn.Module):
    """CompGCN encoder for typed relational edges."""

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_relations: int = 5,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_channels)
        self.rel_embs = nn.Parameter(torch.Tensor(num_relations, hidden_channels))
        nn.init.xavier_uniform_(self.rel_embs)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_channels * 2, hidden_channels))
            
        self.out_proj = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.in_proj(x))
        for layer in self.layers:
            row, col = edge_index
            rel_feat = self.rel_embs[edge_type]
            
            # Message passing with relation embedding
            msg = torch.cat([x[row], rel_feat], dim=-1)
            msg = F.relu(layer(msg))
            
            # Aggregation
            out = torch.zeros_like(x)
            out.index_add_(0, col, msg)
            
            x = x + F.dropout(out, p=self.dropout, training=self.training)
            
        return self.out_proj(x)


class TextEncoder(nn.Module):
    """Mistral-7B-Instruct text encoder with 4-bit QLoRA (r=16)."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", freeze_layers: int = 5):
        super().__init__()

        # 4-bit BitsAndBytes quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # QLoRA adapter: r=16, alpha=32, target q_proj and v_proj
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base_model, lora_config)
        self.hidden_size = self.model.config.hidden_size  # 4096

        logger.info(
            f"TextEncoder: Mistral-7B-Instruct loaded with 4-bit QLoRA "
            f"(r=16, alpha=32). hidden_size={self.hidden_size}"
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Mistral is causal/decoder-only — use the LAST token of the last hidden state
        last_hidden = outputs.hidden_states[-1]  # [B, seq_len, 4096]
        # Gather the position of the last real (non-padded) token per sample
        seq_lengths = attention_mask.sum(dim=-1) - 1  # [B]
        batch_size = last_hidden.size(0)
        last_token_emb = last_hidden[
            torch.arange(batch_size, device=last_hidden.device), seq_lengths
        ]  # [B, 4096]
        return last_token_emb

    def encode_text(self, text: str, tokenizer, device="cuda"):
        """Encode a single text string."""
        self.eval()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            embedding = self.forward(inputs['input_ids'], inputs['attention_mask'])

        return embedding.squeeze(0)


class FusionMLP(nn.Module):
    """Fusion layer combining graph and text embeddings."""

    def __init__(
        self,
        graph_dim: int = 128,
        text_dim: int = 4096,
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

    Combines Hypergraph HGNN + CompGCN + CTDGA (user network) + Mistral-7B QLoRA (text) + Fusion MLP.
    """

    def __init__(
        self,
        graph_in_channels: int = 6,
        graph_hidden: int = 128,
        graph_out: int = 128,
        graph_layers: int = 2,
        text_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        text_freeze: int = 5,
        fusion_hidden: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.hypergraph_encoder = HypergraphHGNN(
            in_channels=graph_in_channels,
            hidden_channels=graph_hidden,
            out_channels=graph_out,
            num_layers=graph_layers,
            dropout=dropout,
        )
        
        self.compgcn_encoder = CompGCNEncoder(
            in_channels=graph_in_channels,
            hidden_channels=graph_hidden,
            out_channels=graph_out,
            num_relations=5,
            num_layers=graph_layers,
            dropout=dropout,
        )
        
        self.ctdga_encoder = CTDGAHawkesEncoder(
            node_dim=graph_in_channels,
            time_dim=32,
            num_heads=4
        )
        
        # Project combined embeddings (Hypergraph + CompGCN + CTDGA)
        self.graph_proj = nn.Linear(graph_out * 2 + graph_in_channels, graph_out)

        self.text_encoder = TextEncoder(
            model_name=text_model,
            freeze_layers=text_freeze,
        )

        self.fusion = FusionMLP(
            graph_dim=graph_out,
            text_dim=4096,
            hidden_dim=fusion_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(
        self,
        x,
        edge_index,
        input_ids,
        attention_mask,
        user_indices,
        edge_weight=None,
        hyperedge_index=None,
        edge_type=None,
        neighbor_embs=None,
        time_deltas=None
    ):
        """Forward pass for batch of (user, text) pairs.

        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Graph edges [2, num_edges]
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            user_indices: Index of user node for each sample [batch_size]
            edge_weight: Optional edge weights [num_edges]
            hyperedge_index: Hypergraph edges [2, num_incidence_pairs]
            edge_type: Edge relation types [num_edges]
            neighbor_embs: Neighbor embeddings for CTDGA [num_nodes, num_neighbors, node_feat_dim]
            time_deltas: Time deltas for CTDGA [num_nodes, num_neighbors]

        Returns:
            logits: [batch_size, num_classes]
        """
        # Dummy fallbacks for testing/inference without full dynamic inputs
        if hyperedge_index is None:
            hyperedge_index = edge_index
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        if neighbor_embs is None:
            # Use self as dummy neighbor
            neighbor_embs = x.unsqueeze(1)
        if time_deltas is None:
            time_deltas = torch.zeros(x.size(0), neighbor_embs.size(1), device=x.device)

        # 1. Hypergraph HGNN
        hg_emb = self.hypergraph_encoder(x, hyperedge_index, edge_weight)
        
        # 2. CompGCN
        cg_emb = self.compgcn_encoder(x, edge_index, edge_type)
        
        # 3. CTDGA Hawkes
        ct_emb = self.ctdga_encoder(x, neighbor_embs, time_deltas)

        # Combine graph embeddings
        combined_graph = torch.cat([hg_emb, cg_emb, ct_emb], dim=-1)
        graph_embeddings = self.graph_proj(combined_graph)

        # Select user embeddings for batch
        user_embeddings = graph_embeddings[user_indices]  # [batch_size, graph_out]

        # Text embeddings
        text_embeddings = self.text_encoder(input_ids, attention_mask)  # [batch_size, 4096]

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
        hyperedge_index=None,
        edge_type=None,
        neighbor_embs=None,
        time_deltas=None,
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
            max_length=1024
        ).to(device)

        # Move graph to device
        x = x.to(device)
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        if hyperedge_index is not None:
            hyperedge_index = hyperedge_index.to(device)
        if edge_type is not None:
            edge_type = edge_type.to(device)
        if neighbor_embs is not None:
            neighbor_embs = neighbor_embs.to(device)
        if time_deltas is not None:
            time_deltas = time_deltas.to(device)

        with torch.no_grad():
            logits = self.forward(
                x=x,
                edge_index=edge_index,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                user_indices=torch.tensor([user_idx], dtype=torch.long, device=device),
                edge_weight=edge_weight,
                hyperedge_index=hyperedge_index,
                edge_type=edge_type,
                neighbor_embs=neighbor_embs,
                time_deltas=time_deltas
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
        hyperedge_index=None,
        edge_type=None,
        neighbor_embs=None,
        time_deltas=None,
        device="cuda",
    ) -> dict:
        """Get intermediate embeddings for interpretability."""
        self.eval()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024
        ).to(device)

        x = x.to(device)
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        if hyperedge_index is not None:
            hyperedge_index = hyperedge_index.to(device)
        if edge_type is not None:
            edge_type = edge_type.to(device)
        if neighbor_embs is not None:
            neighbor_embs = neighbor_embs.to(device)
        if time_deltas is not None:
            time_deltas = time_deltas.to(device)

        with torch.no_grad():
            if hyperedge_index is None:
                hyperedge_index = edge_index
            if edge_type is None:
                edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
            if neighbor_embs is None:
                neighbor_embs = x.unsqueeze(1)
            if time_deltas is None:
                time_deltas = torch.zeros(x.size(0), neighbor_embs.size(1), device=x.device)
                
            hg_emb = self.hypergraph_encoder(x, hyperedge_index, edge_weight)
            cg_emb = self.compgcn_encoder(x, edge_index, edge_type)
            ct_emb = self.ctdga_encoder(x, neighbor_embs, time_deltas)
            
            combined_graph = torch.cat([hg_emb, cg_emb, ct_emb], dim=-1)
            graph_embeddings = self.graph_proj(combined_graph)

            user_emb = graph_embeddings[user_idx]
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
    print("Graph components: HGNN, CompGCN, CTDGA")
    print("Fusion MLP:", model.fusion)
