#!/usr/bin/env python3
"""Main training script for multimodal Doom Index model.

Usage:
    # Single GPU
    python train_multimodal.py --data_path data/processed_reddit.csv --epochs 10

    # Multi-GPU DDP on H100s
    torchrun --nproc_per_node=4 train_multimodal.py --data_path data/processed_reddit.csv --epochs 10 --ddp

    # HPC (PBS)
    qsub hpc_multimodal_train.sh
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.features.graph_extractor import GraphExtractor
from src.models.gnn_model import MultimodalDoomPredictor
from src.models.multimodal_trainer import (
    MultimodalTrainer, DoomDataset, setup_ddp, cleanup_ddp
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(csv_path: str, min_graph_users: int = 1000):
    """Load CSV and prepare train/val splits with user indices."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required = ['text', 'author_id', 'label']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Extract unique users and assign indices
    unique_users = df['author_id'].unique()
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    df['user_idx'] = df['author_id'].map(user_to_idx)

    logger.info(f"Dataset: {len(df)} samples, {len(unique_users)} unique users")

    # Train/val split (stratified by label)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    logger.info(f"Train label distribution: {train_df['label'].value_counts().to_dict()}")

    return train_df, val_df, user_to_idx


def build_graph(user_to_idx: dict, neo4j_uri: str = None):
    """Build or extract graph from Neo4j."""
    logger.info("Building user interaction graph...")

    try:
        extractor = GraphExtractor()
        graph_data, user_df = extractor.extract_user_graph(max_users=50000)

        # Remap graph user IDs to our dataset indices
        # This is a simplified mapping — in production, ensure IDs match
        logger.info(f"Graph extracted: {graph_data.num_nodes} nodes")
        return graph_data

    except Exception as e:
        logger.warning(f"Neo4j extraction failed: {e}. Creating synthetic graph.")
        return create_synthetic_graph(len(user_to_idx))


def create_synthetic_graph(num_users: int):
    """Create a random graph when Neo4j is unavailable."""
    import torch
    from torch_geometric.data import Data

    logger.info(f"Creating synthetic graph with {num_users} users")

    # Random features
    x = torch.randn(num_users, 6)

    # Random edges (Erdos-Renyi-like)
    num_edges = min(num_users * 5, 100000)
    edge_index = torch.randint(0, num_users, (2, num_edges))

    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    data = Data(x=x, edge_index=edge_index, num_nodes=num_users)
    return data


def main():
    parser = argparse.ArgumentParser(description="Train multimodal Doom Index model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to processed CSV")
    parser.add_argument("--output_dir", type=str, default="models/multimodal_doom", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--graph_hidden", type=int, default=128, help="Graph hidden dim")
    parser.add_argument("--graph_layers", type=int, default=2, help="GraphSAGE layers")
    parser.add_argument("--fusion_hidden", type=int, default=256, help="Fusion MLP hidden dim")
    parser.add_argument("--freeze_bert_layers", type=int, default=5, help="Freeze bottom N BERT layers")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=512, help="Max text length")

    args = parser.parse_args()

    # Setup DDP if requested
    local_rank = 0
    if args.ddp:
        local_rank, world_size = setup_ddp()

    try:
        # 1. Load data
        train_df, val_df, user_to_idx = load_and_prepare_data(args.data_path)

        # 2. Build graph
        graph_data = build_graph(user_to_idx)

        # Ensure graph has enough nodes for our users
        if graph_data.num_nodes < len(user_to_idx):
            logger.warning("Graph has fewer nodes than users. Padding with random features.")
            # Pad graph
            num_missing = len(user_to_idx) - graph_data.num_nodes
            pad_x = torch.randn(num_missing, graph_data.x.shape[1])
            graph_data.x = torch.cat([graph_data.x, pad_x], dim=0)
            graph_data.num_nodes = len(user_to_idx)

        # 3. Create model
        model = MultimodalDoomPredictor(
            graph_in_channels=graph_data.x.shape[1],
            graph_hidden=args.graph_hidden,
            graph_out=128,
            graph_layers=args.graph_layers,
            text_model="distilbert-base-uncased",
            text_freeze=args.freeze_bert_layers,
            fusion_hidden=args.fusion_hidden,
            num_classes=2,
            dropout=0.3,
        )

        if local_rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

        # 4. Create datasets
        tokenizer = model.tokenizer

        train_dataset = DoomDataset(
            texts=train_df['text'].tolist(),
            user_indices=train_df['user_idx'].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

        val_dataset = DoomDataset(
            texts=val_df['text'].tolist(),
            user_indices=val_df['user_idx'].tolist(),
            labels=val_df['label'].tolist(),
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

        # 5. Train
        trainer = MultimodalTrainer(
            model=model,
            graph_data=graph_data,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            grad_accum_steps=args.grad_accum,
            fp16=args.fp16,
            ddp=args.ddp,
            local_rank=local_rank,
        )

        trainer.train()

        if local_rank == 0:
            logger.info(f"Training complete. Best model saved to {args.output_dir}/best_model.pt")

    finally:
        if args.ddp:
            cleanup_ddp()


if __name__ == "__main__":
    main()
