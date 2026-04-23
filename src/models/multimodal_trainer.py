"""Multimodal trainer with Distributed Data Parallel (DDP) support.

Optimized for H100 clusters. Supports:
- Mixed precision (FP16/BF16) training
- Gradient accumulation
- Checkpointing
- WandB logging (optional)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm
import numpy as np

from src.models.gnn_model import MultimodalDoomPredictor

logger = logging.getLogger(__name__)


class DoomDataset(Dataset):
    """Dataset for multimodal (graph + text) training.

    Each sample: (user_idx, text, label)
    The graph (x, edge_index) is shared across all samples.
    """

    def __init__(self, texts, user_indices, labels, tokenizer, max_length=512):
        self.texts = texts
        self.user_indices = user_indices
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        user_idx = self.user_indices[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
        }


class MultimodalTrainer:
    """Trainer for multimodal doom predictor."""

    def __init__(
        self,
        model: MultimodalDoomPredictor,
        graph_data,  # PyG Data object
        train_dataset: DoomDataset,
        val_dataset: DoomDataset,
        output_dir: str = "models/multimodal_doom",
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        epochs: int = 10,
        warmup_steps: int = 500,
        grad_accum_steps: int = 2,
        fp16: bool = True,
        ddp: bool = False,
        local_rank: int = 0,
    ):
        self.model = model
        self.graph_data = graph_data
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.grad_accum_steps = grad_accum_steps
        self.fp16 = fp16
        self.ddp = ddp
        self.local_rank = local_rank

        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # Setup data loaders
        self._setup_dataloaders()

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        total_steps = len(self.train_loader) * epochs // grad_accum_steps
        self.scheduler = self._create_scheduler(total_steps)

        # Mixed precision
        self.scaler = GradScaler() if fp16 else None

        # Best model tracking
        self.best_val_f1 = 0.0

        # Move model to device
        self.model.to(self.device)
        self.graph_data = self.graph_data.to(self.device)

        # DDP wrap
        if ddp:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,  # Needed because text encoder has frozen layers
            )

    def _setup_dataloaders(self):
        """Create train/val dataloaders."""
        if self.ddp:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
            )
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
            )
        else:
            train_sampler = None
            val_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,  # Larger batch for val
            sampler=val_sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def _create_optimizer(self):
        """Create AdamW with weight decay."""
        no_decay = ['bias', 'LayerNorm.weight']

        # Separate parameters: graph, text, fusion
        graph_params = []
        text_params = []
        fusion_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'graph_encoder' in name:
                graph_params.append((name, param))
            elif 'text_encoder' in name:
                text_params.append((name, param))
            else:
                fusion_params.append((name, param))

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in graph_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay,
                'lr': self.learning_rate * 2,  # Graph can use higher LR
            },
            {
                'params': [p for n, p in graph_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate * 2,
            },
            {
                'params': [p for n, p in text_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay,
                'lr': self.learning_rate,
            },
            {
                'params': [p for n, p in text_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate,
            },
            {
                'params': [p for n, p in fusion_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay,
                'lr': self.learning_rate * 3,  # Fusion head trains faster
            },
            {
                'params': [p for n, p in fusion_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate * 3,
            },
        ]

        return torch.optim.AdamW(optimizer_grouped_parameters)

    def _create_scheduler(self, total_steps):
        """Create linear warmup + linear decay scheduler."""
        from transformers import get_linear_schedule_with_warmup

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()

        if self.ddp:
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=self.local_rank != 0)

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            user_indices = batch['user_idx'].to(self.device)
            labels = batch['label'].to(self.device)

            with autocast(enabled=self.fp16):
                logits = self.model(
                    x=self.graph_data.x,
                    edge_index=self.graph_data.edge_index,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    user_indices=user_indices,
                    edge_weight=getattr(self.graph_data, 'edge_weight', None),
                )

                loss = nn.functional.cross_entropy(logits, labels)
                loss = loss / self.grad_accum_steps

            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

            if self.local_rank == 0:
                progress.set_postfix({
                    'loss': f"{total_loss / num_batches:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc="Evaluating", disable=self.local_rank != 0):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            user_indices = batch['user_idx'].to(self.device)
            labels = batch['label'].to(self.device)

            with autocast(enabled=self.fp16):
                logits = self.model(
                    x=self.graph_data.x,
                    edge_index=self.graph_data.edge_index,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    user_indices=user_indices,
                    edge_weight=getattr(self.graph_data, 'edge_weight', None),
                )

                loss = nn.functional.cross_entropy(logits, labels)

            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        avg_loss = total_loss / len(self.val_loader)

        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': acc,
            'val_f1': f1,
            'val_auc': auc,
        }

        return metrics, all_labels, all_preds

    def train(self):
        """Full training loop."""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        logger.info(f"Graph: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)

            if self.ddp:
                dist.barrier()

            # Only evaluate on rank 0
            if self.local_rank == 0:
                metrics, labels, preds = self.evaluate()

                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={metrics['val_loss']:.4f}, "
                    f"val_acc={metrics['val_accuracy']:.4f}, "
                    f"val_f1={metrics['val_f1']:.4f}, "
                    f"val_auc={metrics['val_auc']:.4f}"
                )

                # Save best model
                if metrics['val_f1'] > self.best_val_f1:
                    self.best_val_f1 = metrics['val_f1']
                    self.save_checkpoint(epoch, metrics, is_best=True)

                # Save periodic checkpoint
                if epoch % 2 == 0:
                    self.save_checkpoint(epoch, metrics, is_best=False)

        if self.local_rank == 0:
            logger.info(f"Training complete. Best val F1: {self.best_val_f1:.4f}")

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        # Unwrap DDP if needed
        model_to_save = self.model.module if self.ddp else self.model

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
        }

        if is_best:
            path = self.output_dir / "best_model.pt"
            logger.info(f"Saving best model to {path}")
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)

        # Save config
        config_path = self.output_dir / "model_config.pt"
        if not config_path.exists():
            torch.save({
                'graph_in_channels': 6,
                'graph_hidden': 128,
                'graph_out': 128,
                'graph_layers': 2,
                'text_model': "distilbert-base-uncased",
                'fusion_hidden': 256,
            }, config_path)


def setup_ddp():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
        )

        torch.cuda.set_device(local_rank)
        logger.info(f"DDP initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return local_rank, world_size
    else:
        return 0, 1


def cleanup_ddp():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # Example usage
    print("Multimodal trainer module. Use train_multimodal.py to run.")
