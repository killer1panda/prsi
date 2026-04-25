#!/usr/bin/env python3
"""
Text-Only Baseline Training for Doom Index v2
Uses DistilBERT for text classification with doom score regression
Optimized for HPC H100 cluster
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score
from tqdm import tqdm
import json
from datetime import datetime

class DoomTextDataset(Dataset):
    def __init__(self, texts, doom_scores, tokenizer, max_length=128):
        self.texts = texts
        self.doom_scores = doom_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'doom_score': torch.tensor(self.doom_scores[idx], dtype=torch.float32)
        }

class DoomDistilBert(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(pooled)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze()

def find_latest_parquet(data_dir: str) -> tuple:
    """Find the most recent train/val/test parquet files"""
    data_path = Path(data_dir)
    
    train_files = sorted(data_path.glob('train_*.parquet'))
    val_files = sorted(data_path.glob('val_*.parquet'))
    test_files = sorted(data_path.glob('test_*.parquet'))
    
    if not train_files or not val_files or not test_files:
        raise ValueError(f"No processed parquet files found in {data_dir}. Run consolidate_production.py first.")
    
    return str(train_files[-1]), str(val_files[-1]), str(test_files[-1])

def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['doom_score'].to(device)
        
        optimizer.zero_grad()
        preds = model(input_ids, attention_mask)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    return total_loss / len(dataloader), all_preds, all_targets

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['doom_score'].to(device)
            
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, targets)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return total_loss / len(dataloader), all_preds, all_targets

def calculate_metrics(preds, targets):
    preds_np = np.array(preds)
    targets_np = np.array(targets)
    
    # Convert to binary for classification metrics (threshold=0.5)
    preds_binary = (preds_np > 0.5).astype(int)
    targets_binary = (targets_np > 0.5).astype(int)
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(targets_np, preds_np))),
        'mae': float(mean_absolute_error(targets_np, preds_np)),
        'accuracy': float(accuracy_score(targets_binary, preds_binary)),
    }
    
    # AUC only if both classes present
    if len(np.unique(targets_binary)) > 1:
        metrics['auc'] = float(roc_auc_score(targets_binary, preds_np))
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Train Doom Index Text Baseline')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Processed data directory')
    parser.add_argument('--output-dir', type=str, default='models/text_baseline', help='Model output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find latest data files
    train_path, val_path, test_path = find_latest_parquet(args.data_dir)
    print(f"\nLoading data:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")
    
    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Datasets
    train_dataset = DoomTextDataset(
        train_df['text'].tolist(),
        train_df['doom_score'].tolist(),
        tokenizer,
        args.max_length
    )
    val_dataset = DoomTextDataset(
        val_df['text'].tolist(),
        val_df['doom_score'].tolist(),
        tokenizer,
        args.max_length
    )
    test_dataset = DoomTextDataset(
        test_df['text'].tolist(),
        test_df['doom_score'].tolist(),
        tokenizer,
        args.max_length
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = DoomDistilBert().to(device)
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*60)
    
    best_val_loss = float('inf')
    history = {'train': [], 'val': [], 'test': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_preds, train_targets = train_epoch(model, train_loader, optimizer, device, criterion)
        scheduler.step()
        
        # Validate
        val_loss, val_preds, val_targets = evaluate(model, val_loader, device, criterion)
        
        # Test
        test_loss, test_preds, test_targets = evaluate(model, test_loader, device, criterion)
        
        # Metrics
        train_metrics = calculate_metrics(train_preds, train_targets)
        val_metrics = calculate_metrics(val_preds, val_targets)
        test_metrics = calculate_metrics(test_preds, test_targets)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | RMSE: {train_metrics['rmse']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | RMSE: {val_metrics['rmse']:.4f} | Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics.get('auc', 'N/A')}")
        print(f"  Test  Loss: {test_loss:.4f} | RMSE: {test_metrics['rmse']:.4f} | Acc: {test_metrics['accuracy']:.4f} | AUC: {test_metrics.get('auc', 'N/A')}")
        
        history['train'].append({'loss': train_loss, **train_metrics})
        history['val'].append({'loss': val_loss, **val_metrics})
        history['test'].append({'loss': test_loss, **test_metrics})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, output_path / 'best_model.pt')
            print(f"  ✅ Saved best model (val_loss={val_loss:.4f})")
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_metrics': val_metrics,
    }, output_path / 'final_model.pt')
    
    # Save history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    config['final_metrics'] = test_metrics
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best model saved to: {output_path / 'best_model.pt'}")
    print(f"Final test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nReady for viva demo!")

if __name__ == '__main__':
    main()
