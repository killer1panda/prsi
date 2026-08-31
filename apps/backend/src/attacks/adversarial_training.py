"""
Adversarial Training module: generates adversarial examples during training
and trains on them to improve model robustness against shadowban attacks.
"""
import logging
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class AdvTrainingConfig:
    epsilon: float = 0.3  # PGD perturbation budget
    alpha: float = 0.01   # PGD step size
    num_steps: int = 5    # PGD iterations
    adv_ratio: float = 0.5  # Ratio of adversarial examples in batch
    mixup_alpha: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PGDAttack:
    """Projected Gradient Descent for generating adversarial text embeddings."""

    def __init__(self, model: nn.Module, config: AdvTrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

    def generate(self, embeddings: torch.Tensor, labels: torch.Tensor,
                 loss_fn: Callable) -> torch.Tensor:
        """
        Generate adversarial embeddings using PGD.

        Args:
            embeddings: (B, D) input embeddings
            labels: (B,) true labels
            loss_fn: Loss function
        Returns:
            (B, D) adversarial embeddings
        """
        delta = torch.zeros_like(embeddings, requires_grad=True)

        for _ in range(self.config.num_steps):
            if delta.grad is not None:
                delta.grad.zero_()

            adv_emb = embeddings + delta
            outputs = self.model(adv_emb)
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Gradient ascent on delta
            delta.data = delta.data + self.config.alpha * delta.grad.sign()
            # Project to epsilon ball
            delta.data = torch.clamp(delta.data, -self.config.epsilon, self.config.epsilon)
            # Ensure embeddings + delta stay in valid range (approximate)
            delta.data = torch.clamp(embeddings + delta.data, -3.0, 3.0) - embeddings

        return embeddings + delta.detach()


class AdversarialTrainer:
    """
    Adversarial training loop that mixes clean and adversarial batches.
    Improves robustness against TextAttack-style perturbations.
    """

    def __init__(self, model: nn.Module, config: Optional[AdvTrainingConfig] = None):
        self.model = model
        self.config = config or AdvTrainingConfig()
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)
        self.pgd = PGDAttack(model, self.config)
        logger.info("AdversarialTrainer initialized")

    def mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Mixup augmentation for better generalization."""
        lam = torch.distributions.Beta(self.config.mixup_alpha, self.config.mixup_alpha).sample().item()
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      optimizer: torch.optim.Optimizer,
                      loss_fn: Callable) -> Dict[str, float]:
        """
        Single adversarial training step.

        Returns:
            Dict with losses and metrics
        """
        self.model.train()
        optimizer.zero_grad()

        embeddings, labels = batch
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)

        # Split batch: clean vs adversarial
        batch_size = embeddings.size(0)
        adv_size = int(batch_size * self.config.adv_ratio)

        clean_emb = embeddings[adv_size:]
        clean_labels = labels[adv_size:]

        adv_emb = embeddings[:adv_size]
        adv_labels = labels[:adv_size]

        total_loss = 0.0

        # Clean forward
        if clean_emb.size(0) > 0:
            if self.config.mixup_alpha > 0:
                mixed_x, y_a, y_b, lam = self.mixup(clean_emb, clean_labels)
                outputs = self.model(mixed_x)
                loss_clean = lam * loss_fn(outputs, y_a) + (1 - lam) * loss_fn(outputs, y_b)
            else:
                outputs = self.model(clean_emb)
                loss_clean = loss_fn(outputs, clean_labels)
            total_loss += loss_clean

        # Adversarial forward
        if adv_emb.size(0) > 0:
            adv_emb_pgd = self.pgd.generate(adv_emb, adv_labels, loss_fn)
            outputs_adv = self.model(adv_emb_pgd)
            loss_adv = loss_fn(outputs_adv, adv_labels)
            total_loss += loss_adv

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute accuracy on clean data
        with torch.no_grad():
            preds = torch.sigmoid(self.model(embeddings)).squeeze()
            acc = ((preds > 0.5).float() == labels.float()).float().mean().item()

        return {
            "loss": total_loss.item(),
            "accuracy": acc,
            "clean_batch_size": clean_emb.size(0),
            "adv_batch_size": adv_emb.size(0)
        }

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader],
            optimizer: torch.optim.Optimizer, loss_fn: Callable,
            num_epochs: int = 10) -> Dict[str, List[float]]:
        """Full adversarial training loop."""
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0

            for batch in train_loader:
                metrics = self.training_step(batch, optimizer, loss_fn)
                epoch_loss += metrics["loss"]
                epoch_acc += metrics["accuracy"]
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_acc = epoch_acc / max(num_batches, 1)
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(avg_acc)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, loss_fn)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} Acc: {avg_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")

        return history

    def evaluate(self, data_loader: DataLoader, loss_fn: Callable) -> Tuple[float, float]:
        """Evaluate on clean data."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for embeddings, labels in data_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(embeddings)
                loss = loss_fn(outputs, labels)
                preds = torch.sigmoid(outputs).squeeze()
                acc = ((preds > 0.5).float() == labels.float()).float().mean().item()

                total_loss += loss.item()
                total_acc += acc
                num_batches += 1

        return total_loss / max(num_batches, 1), total_acc / max(num_batches, 1)

    def robustness_test(self, attack_fn: Callable, data_loader: DataLoader) -> float:
        """Test robustness against a given attack function."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for embeddings, labels in data_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                adv_emb = attack_fn(embeddings, labels)
                outputs = self.model(adv_emb)
                preds = (torch.sigmoid(outputs).squeeze() > 0.5).float()
                correct += (preds == labels.float()).sum().item()
                total += labels.size(0)

        return correct / max(total, 1)
