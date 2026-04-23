"""Differential Privacy training for Doom Index.

Uses Opacus to add Gaussian noise during training, providing
(ε, δ)-differential privacy guarantees.
"""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from src.models.gnn_model import MultimodalDoomPredictor
from src.models.multimodal_trainer import DoomDataset

logger = logging.getLogger(__name__)


class DPDoomTrainer:
    """Differentially private trainer for multimodal doom predictor.

    Only the text encoder and fusion head are trained with DP.
    The graph encoder is frozen (graph structure is public knowledge).
    """

    def __init__(
        self,
        model: MultimodalDoomPredictor,
        graph_data,
        train_dataset: DoomDataset,
        val_dataset: DoomDataset,
        output_dir: str = "models/dp_doom",
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        epochs: int = 10,
        # DP params
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.graph_data = graph_data.to(next(model.parameters()).device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        self.device = next(model.parameters()).device

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        # Freeze graph encoder (graph is public, no privacy needed)
        for param in model.graph_encoder.parameters():
            param.requires_grad = False

        # Validate model for DP compatibility
        self.model = ModuleValidator.fix(self.model)

        # Optimizer (only trainable params)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )

        # Privacy engine
        self.privacy_engine = PrivacyEngine()

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )

        logger.info(f"DP Trainer initialized: ε={epsilon}, δ={delta}")
        logger.info(f"Privacy budget per epoch: ~{epsilon/epochs:.2f}")

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch with DP."""
        self.model.train()

        # Unfreeze graph batch norm for train mode (but not weights)
        for module in self.model.graph_encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.train()

        total_loss = 0.0
        num_batches = 0

        progress = tqdm(self.train_loader, desc=f"DP Epoch {epoch}")

        for batch in progress:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            user_indices = batch['user_idx'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(
                x=self.graph_data.x,
                edge_index=self.graph_data.edge_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                user_indices=user_indices,
            )

            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Get current epsilon
            eps = self.privacy_engine.get_epsilon(self.delta)
            progress.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'ε': f"{eps:.2f}",
            })

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict, list, list]:
        """Evaluate (no DP noise in eval)."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc="DP Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            user_indices = batch['user_idx'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.model(
                x=self.graph_data.x,
                edge_index=self.graph_data.edge_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                user_indices=user_indices,
            )

            loss = nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        final_epsilon = self.privacy_engine.get_epsilon(self.delta)

        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_accuracy': acc,
            'val_f1': f1,
            'val_auc': auc,
            'epsilon': final_epsilon,
            'delta': self.delta,
        }

        return metrics, all_labels, all_preds

    def train(self):
        """Full DP training loop."""
        logger.info("Starting differentially private training...")
        best_f1 = 0.0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            metrics, _, _ = self.evaluate()

            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_acc={metrics['val_accuracy']:.4f}, "
                f"val_f1={metrics['val_f1']:.4f}, "
                f"ε={metrics['epsilon']:.2f}"
            )

            if metrics['val_f1'] > best_f1:
                best_f1 = metrics['val_f1']
                self.save_checkpoint(epoch, metrics, is_best=True)

        logger.info(f"DP Training complete. Best F1: {best_f1:.4f}, Final ε: {metrics['epsilon']:.2f}")

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save checkpoint."""
        model_to_save = self.model._module if hasattr(self.model, '_module') else self.model

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'metrics': metrics,
            'privacy': {
                'epsilon': metrics['epsilon'],
                'delta': metrics['delta'],
                'max_grad_norm': self.max_grad_norm,
            }
        }

        if is_best:
            path = self.output_dir / f"best_dp_e{self.epsilon}.pt"
            logger.info(f"Saving best DP model to {path}")
            torch.save(checkpoint, path)


def run_dp_experiments(
    model,
    graph_data,
    train_dataset,
    val_dataset,
    epsilons=[0.1, 0.5, 1.0, 2.0, 5.0, float('inf')],
    output_dir="models/dp_experiments",
):
    """Run DP training with multiple privacy budgets and return tradeoff data."""
    results = []

    for eps in epsilons:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training with ε = {eps}")
        logger.info(f"{'='*50}")

        if eps == float('inf'):
            # Non-private baseline
            from src.models.multimodal_trainer import MultimodalTrainer
            trainer = MultimodalTrainer(
                model=model,
                graph_data=graph_data,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir=f"{output_dir}/eps_inf",
                epochs=5,
            )
            trainer.train()
            metrics = {'val_accuracy': trainer.best_val_f1, 'val_f1': trainer.best_val_f1, 
                      'epsilon': float('inf'), 'delta': 0}
        else:
            trainer = DPDoomTrainer(
                model=model,
                graph_data=graph_data,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir=f"{output_dir}/eps_{eps}",
                epsilon=eps,
                epochs=5,
            )
            trainer.train()
            metrics, _, _ = trainer.evaluate()

        results.append({
            'epsilon': eps,
            'accuracy': metrics['val_accuracy'],
            'f1': metrics['val_f1'],
            'auc': metrics.get('val_auc', 0),
        })

    # Save tradeoff data
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/privacy_utility_tradeoff.csv", index=False)
    logger.info(f"Tradeoff data saved to {output_dir}/privacy_utility_tradeoff.csv")

    return df


if __name__ == "__main__":
    print("DP Trainer module. Use train_multimodal.py with --dp flag to run.")
