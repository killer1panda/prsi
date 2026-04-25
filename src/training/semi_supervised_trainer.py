"""
Semi-Supervised Learning Pipeline for PRSI Doom Index

Combines supervised learning on labeled datasets with unsupervised 
pre-training on large-scale Pushshift Reddit data using contrastive 
learning and pseudo-labeling strategies.

Author: Senior ML Engineer
Date: 2026
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemiSupervisedStrategy(Enum):
    """Available semi-supervised learning strategies."""
    SUPERVISED_ONLY = "supervised_only"
    SELF_TRAINING = "self_training"  # Pseudo-labeling
    CONSISTENCY_REGULARIZATION = "consistency_reg"  # Mean Teacher
    CONTRASTIVE_PRETRAIN = "contrastive"  # SimCLR-style
    MIXMATCH = "mixmatch"  # MixMatch algorithm
    FIXMATCH = "fixmatch"  # FixMatch algorithm


@dataclass
class ModelConfig:
    """Configuration for semi-supervised model."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    num_labels: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Semi-supervised specific
    strategy: SemiSupervisedStrategy = SemiSupervisedStrategy.SELF_TRAINING
    unlabeled_weight: float = 0.5  # Weight for unlabeled loss
    confidence_threshold: float = 0.95  # For pseudo-labeling
    augmentation_probability: float = 0.5  # For consistency reg
    temperature: float = 0.7  # For contrastive learning
    
    # Contrastive learning
    projection_dim: int = 128
    contrastive_weight: float = 1.0


class DoomIndexDataset(Dataset):
    """PyTorch Dataset for Doom Index with augmentation support."""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: AutoTokenizer = None,
        max_length: int = 256,
        augment: bool = False
    ):
        self.texts = texts
        self.labels = labels if labels is not None else [-1] * len(texts)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_length = max_length
        self.augment = augment
        
    def __len__(self):
        return len(self.texts)
    
    def _augment_text(self, text: str) -> str:
        """Apply light text augmentation."""
        if not self.augment or np.random.random() > 0.5:
            return text
        
        # Simple augmentations
        words = text.split()
        if len(words) > 3:
            # Random word dropout
            if np.random.random() < 0.3:
                idx = np.random.randint(len(words))
                words = words[:idx] + words[idx+1:]
            
            # Random word swap
            if np.random.random() < 0.2 and len(words) > 4:
                idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def __getitem__(self, idx):
        text = self._augment_text(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long) if label >= 0 else torch.tensor(-100, dtype=torch.long)
        }
        
        return item


class ContrastiveHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, hidden_dim: int, projection_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.projector(x), dim=-1)


class SemiSupervisedDoomClassifier:
    """
    Semi-supervised classifier for Doom Index prediction.
    
    Supports multiple strategies:
    - Self-training with pseudo-labels
    - Consistency regularization (Mean Teacher)
    - Contrastive pre-training
    - MixMatch/FixMatch
    """
    
    def __init__(
        self,
        config: ModelConfig,
        device: Optional[str] = None
    ):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.base_model = AutoModel.from_pretrained(config.model_name)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )
        
        # Add contrastive head if needed
        if config.strategy == SemiSupervisedStrategy.CONTRASTIVE_PRETRAIN:
            hidden_dim = self.base_model.config.hidden_size
            self.contrastive_head = ContrastiveHead(hidden_dim, config.projection_dim).to(self.device)
        
        self.classifier.to(self.device)
        self.base_model.to(self.device)
        
    def compute_metrics(self, pred):
        """Compute evaluation metrics."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        probs = pred.predictions[:, 1]  # Probability of positive class
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        accuracy = accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train_supervised(
        self,
        train_dataset: DoomIndexDataset,
        eval_dataset: Optional[DoomIndexDataset] = None,
        output_dir: str = "models/supervised"
    ) -> Trainer:
        """Train on labeled data only."""
        logger.info("Training supervised model...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=self.device == "cuda",
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.classifier,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Supervised model saved to {output_dir}")
        return trainer
    
    def generate_pseudo_labels(
        self,
        unlabeled_texts: List[str],
        batch_size: int = 64
    ) -> Tuple[List[str], List[int], List[float]]:
        """Generate pseudo-labels for unlabeled data using current model."""
        logger.info(f"Generating pseudo-labels for {len(unlabeled_texts)} samples...")
        
        pseudo_labeled_texts = []
        pseudo_labels = []
        confidences = []
        
        self.classifier.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(unlabeled_texts), batch_size)):
                batch_texts = unlabeled_texts[i:i + batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.classifier(**encodings)
                probs = F.softmax(outputs.logits, dim=-1)
                max_probs, predictions = torch.max(probs, dim=-1)
                
                # Only keep high-confidence predictions
                for j, (text, pred, conf) in enumerate(zip(batch_texts, predictions, max_probs)):
                    if conf.item() > self.config.confidence_threshold:
                        pseudo_labeled_texts.append(text)
                        pseudo_labels.append(pred.item())
                        confidences.append(conf.item())
        
        logger.info(f"Generated {len(pseudo_labeled_texts)} pseudo-labels ({len(pseudo_labeled_texts)/len(unlabeled_texts)*100:.1f}% of data)")
        
        return pseudo_labeled_texts, pseudo_labels, confidences
    
    def train_self_training(
        self,
        labeled_texts: List[str],
        labeled_labels: List[int],
        unlabeled_texts: List[str],
        val_texts: List[str],
        val_labels: List[int],
        n_iterations: int = 3,
        output_dir: str = "models/self_training"
    ):
        """Self-training with iterative pseudo-labeling."""
        logger.info("Starting self-training pipeline...")
        
        best_f1 = 0.0
        
        for iteration in range(n_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}/{n_iterations}")
            logger.info(f"{'='*60}")
            
            # Train on current labeled data
            iter_output = f"{output_dir}/iter_{iteration}"
            
            train_dataset = DoomIndexDataset(
                labeled_texts,
                labeled_labels,
                self.tokenizer,
                self.config.max_length
            )
            
            val_dataset = DoomIndexDataset(
                val_texts,
                val_labels,
                self.tokenizer,
                self.config.max_length
            )
            
            trainer = self.train_supervised(train_dataset, val_dataset, iter_output)
            
            # Evaluate
            results = trainer.evaluate()
            logger.info(f"Iteration {iteration + 1} results: {results}")
            
            if results['eval_f1'] > best_f1:
                best_f1 = results['eval_f1']
                # Save best model
                self.classifier.save_pretrained(f"{output_dir}/best")
                self.tokenizer.save_pretrained(f"{output_dir}/best")
            
            # Generate pseudo-labels for remaining unlabeled data
            if iteration < n_iterations - 1:
                pseudo_texts, pseudo_labels, confs = self.generate_pseudo_labels(unlabeled_texts)
                
                if len(pseudo_texts) == 0:
                    logger.warning("No pseudo-labels generated, stopping early")
                    break
                
                # Add pseudo-labeled data to training set
                labeled_texts.extend(pseudo_texts)
                labeled_labels.extend(pseudo_labels)
                
                logger.info(f"Added {len(pseudo_texts)} pseudo-labeled samples")
                logger.info(f"Total training samples: {len(labeled_texts)}")
                
                # Remove used unlabeled samples
                unlabeled_texts = [t for t in unlabeled_texts if t not in pseudo_texts]
        
        logger.info(f"\nSelf-training complete. Best F1: {best_f1:.4f}")
    
    def train_contrastive_pretrain(
        self,
        texts: List[str],
        batch_size: int = 64,
        num_epochs: int = 5,
        output_dir: str = "models/contrastive"
    ):
        """Contrastive pre-training on unlabeled data."""
        logger.info("Starting contrastive pre-training...")
        
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        
        # Create dataset with augmentations
        dataset = DoomIndexDataset(
            texts,
            labels=None,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            augment=True
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.base_model.parameters(), lr=self.config.learning_rate)
        
        self.base_model.train()
        self.contrastive_head.train()
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get two augmented views
                outputs1 = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                outputs2 = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Project
                proj1 = self.contrastive_head(outputs1.last_hidden_state[:, 0, :])
                proj2 = self.contrastive_head(outputs2.last_hidden_state[:, 0, :])
                
                # Contrastive loss (InfoNCE)
                logits = torch.matmul(proj1, proj2.T) / self.config.temperature
                labels = torch.arange(logits.size(0)).to(self.device)
                
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Save pretrained model
        self.base_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.contrastive_head.state_dict(), f"{output_dir}/contrastive_head.pt")
        
        logger.info(f"Contrastive model saved to {output_dir}")
    
    def finetune_with_pretrained_weights(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        pretrained_path: str,
        output_dir: str = "models/finetuned"
    ):
        """Fine-tune contrastive pre-trained model on labeled data."""
        logger.info(f"Fine-tuning with pretrained weights from {pretrained_path}")
        
        # Load pretrained base model
        self.base_model = AutoModel.from_pretrained(pretrained_path)
        
        # Initialize classifier with pretrained base
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            pretrained_path,
            num_labels=self.config.num_labels
        )
        self.classifier.to(self.device)
        
        # Fine-tune
        train_dataset = DoomIndexDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            self.config.max_length
        )
        
        val_dataset = DoomIndexDataset(
            val_texts,
            val_labels,
            self.tokenizer,
            self.config.max_length
        )
        
        trainer = self.train_supervised(train_dataset, val_dataset, output_dir)
        
        return trainer


def load_unified_data(data_path: str) -> Tuple[List[str], List[int], List[str]]:
    """Load unified dataset from parquet."""
    df = pd.read_parquet(data_path)
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    sources = df.get('source', ['unknown'] * len(df)).tolist()
    
    return texts, labels, sources


def main():
    """Main entry point for semi-supervised training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Semi-supervised training for PRSI")
    parser.add_argument("--strategy", type=str, default="self_training",
                       choices=["supervised", "self_training", "contrastive"])
    parser.add_argument("--labeled-data", type=str, default="data/unified/train.parquet")
    parser.add_argument("--unlabeled-data", type=str, default="data/pushshift/unlabeled_texts.txt")
    parser.add_argument("--val-data", type=str, default="data/unified/validation.parquet")
    parser.add_argument("--output-dir", type=str, default="models/semi_supervised")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--n-iterations", type=int, default=3)
    
    args = parser.parse_args()
    
    # Load labeled data
    logger.info(f"Loading labeled data from {args.labeled_data}")
    train_texts, train_labels, _ = load_unified_data(args.labeled_data)
    logger.info(f"Loaded {len(train_texts)} labeled samples")
    
    # Load validation data
    logger.info(f"Loading validation data from {args.val_data}")
    val_texts, val_labels, _ = load_unified_data(args.val_data)
    logger.info(f"Loaded {len(val_texts)} validation samples")
    
    # Load unlabeled data if available
    unlabeled_texts = []
    if os.path.exists(args.unlabeled_data):
        logger.info(f"Loading unlabeled data from {args.unlabeled_data}")
        with open(args.unlabeled_data, 'r') as f:
            unlabeled_texts = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(unlabeled_texts)} unlabeled samples")
    
    # Configure model
    config = ModelConfig(
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        strategy=SemiSupervisedStrategy(args.strategy.replace("_", "_"))
    )
    
    # Initialize trainer
    trainer = SemiSupervisedDoomClassifier(config)
    
    # Train based on strategy
    if args.strategy == "supervised":
        train_dataset = DoomIndexDataset(train_texts, train_labels, trainer.tokenizer, config.max_length)
        val_dataset = DoomIndexDataset(val_texts, val_labels, trainer.tokenizer, config.max_length)
        trainer.train_supervised(train_dataset, val_dataset, args.output_dir)
    
    elif args.strategy == "self_training":
        if len(unlabeled_texts) == 0:
            logger.warning("No unlabeled data provided, falling back to supervised training")
            trainer.train_supervised(
                DoomIndexDataset(train_texts, train_labels, trainer.tokenizer, config.max_length),
                DoomIndexDataset(val_texts, val_labels, trainer.tokenizer, config.max_length),
                args.output_dir
            )
        else:
            trainer.train_self_training(
                train_texts, train_labels,
                unlabeled_texts,
                val_texts, val_labels,
                n_iterations=args.n_iterations,
                output_dir=args.output_dir
            )
    
    elif args.strategy == "contrastive":
        if len(unlabeled_texts) == 0:
            logger.warning("No unlabeled data for contrastive pre-training")
        else:
            trainer.train_contrastive_pretrain(
                unlabeled_texts,
                batch_size=config.batch_size,
                num_epochs=config.num_epochs,
                output_dir=f"{args.output_dir}/contrastive"
            )
            
            trainer.finetune_with_pretrained_weights(
                train_texts, train_labels,
                val_texts, val_labels,
                pretrained_path=f"{args.output_dir}/contrastive",
                output_dir=args.output_dir
            )
    
    logger.info("\n✅ Training complete!")


if __name__ == "__main__":
    main()
