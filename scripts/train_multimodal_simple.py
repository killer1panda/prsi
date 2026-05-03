#!/usr/bin/env python3
"""
Doom Index v2 - Simple Multimodal Training (Text + Image)
Uses CLIP (Image) + DistilBERT (Text) -> Fusion MLP
Works on single GPU, no complex graph setup required.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import os

# Install CLIP if missing
try:
    import clip
except ImportError:
    print("⚠️ Installing CLIP...")
    os.system("pip install git+https://github.com/openai/CLIP.git")
    import clip

class DoomMultimodalDataset(Dataset):
    def __init__(self, df, tokenizer, clip_preprocess, max_length=128, image_root=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.preprocess = clip_preprocess
        self.max_length = max_length
        self.image_root = Path(image_root) if image_root else None
        
        # Blank image fallback
        blank = Image.new('RGB', (224, 224), color='black')
        self.blank_tensor = clip_preprocess(blank)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row.get('text', ''))
        score = float(row.get('doom_score', 0.5))
        
        # Text
        enc = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_length,
                                         padding='max_length', truncation=True, return_tensors='pt')
        
        # Image
        img_tensor = self.blank_tensor
        img_path = row.get('image_path')
        if img_path and self.image_root:
            full_path = self.image_root / img_path
            if full_path.exists():
                try:
                    img = Image.open(full_path).convert('RGB')
                    img_tensor = self.preprocess(img)
                except: pass
        
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'image': img_tensor,
            'doom_score': torch.tensor(score, dtype=torch.float32)
        }

class MultimodalDoomNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.clip_model, _ = clip.load("ViT-B/32", device='cpu', jit=False)
        self.clip_model.eval() # Freeze CLIP
        
        # Fusion: 768 (BERT) + 512 (CLIP)
        self.fusion = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, image):
        txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_emb = txt_out.last_hidden_state[:, 0, :]
        
        with torch.no_grad():
            img_emb = self.clip_model.encode_image(image.to(input_ids.device)).float()
        
        combined = torch.cat((txt_emb, img_emb), dim=1)
        return self.fusion(combined).squeeze()

def train_epoch(model, loader, opt, dev, crit):
    model.train()
    total_loss, preds, targets = 0, [], []
    for batch in tqdm(loader, desc="Training"):
        ids, mask = batch['input_ids'].to(dev), batch['attention_mask'].to(dev)
        imgs, targs = batch['image'].to(dev), batch['doom_score'].to(dev)
        
        opt.zero_grad()
        outs = model(ids, mask, imgs)
        loss = crit(outs, targs)
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
        preds.extend(outs.cpu().detach().numpy())
        targets.extend(targs.cpu().numpy())
    return total_loss/len(loader), preds, targets

def evaluate(model, loader, dev, crit):
    model.eval()
    total_loss, preds, targets = 0, [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            ids, mask = batch['input_ids'].to(dev), batch['attention_mask'].to(dev)
            imgs, targs = batch['image'].to(dev), batch['doom_score'].to(dev)
            
            outs = model(ids, mask, imgs)
            loss = crit(outs, targs)
            total_loss += loss.item()
            preds.extend(outs.cpu().numpy())
            targets.extend(targs.cpu().numpy())
    return total_loss/len(loader), preds, targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/unified_train_fixed.parquet')
    parser.add_argument('--img-dir', type=str, default='/home/vivek.120542/hateful_memes/data/img')
    parser.add_argument('--out', type=str, default='models/multimodal_v1')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    # Load Data (Filter for Memelens which has images)
    df = pd.read_parquet(args.data)
    if 'source' in df.columns:
        df = df[df['source'] == 'memelens'].reset_index(drop=True)
    
    if len(df) < 100:
        print("❌ Not enough meme data found. Check image paths and source column.")
        return
    
    print(f"✅ Loaded {len(df)} meme samples.")

    # Split
    from sklearn.model_selection import train_test_split
    train_df, temp = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42)
    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Setup
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    _, preprocess = clip.load("ViT-B/32", device='cpu')
    
    train_ds = DoomMultimodalDataset(train_df, tokenizer, preprocess, image_root=args.img_dir)
    val_ds = DoomMultimodalDataset(val_df, tokenizer, preprocess, image_root=args.img_dir)
    test_ds = DoomMultimodalDataset(test_df, tokenizer, preprocess, image_root=args.img_dir)

    loaders = {
        'train': DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2),
        'val': DataLoader(val_ds, batch_size=args.batch),
        'test': DataLoader(test_ds, batch_size=args.batch)
    }

    model = MultimodalDoomNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=args.lr) # Only train fusion + BERT

    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        t_loss, _, _ = train_epoch(model, loaders['train'], optimizer, device, criterion)
        v_loss, v_p, v_t = evaluate(model, loaders['val'], device, criterion)
        
        # Metrics
        v_bin = [1 if x > 0.5 else 0 for x in v_t]
        v_pred = [1 if x > 0.5 else 0 for x in v_p]
        acc = accuracy_score(v_bin, v_pred)
        auc = roc_auc_score(v_bin, v_p) if len(set(v_bin)) > 1 else 0
        
        print(f"Epoch {epoch+1}: Loss={v_loss:.4f}, Acc={acc:.4f}, AUC={auc:.4f}")
        
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save({'model_state_dict': model.state_dict(), 'auc': auc}, 
                       Path(args.out) / 'best_model.pt')
            print("  💾 Saved Best Model")

    # Final Test
    _, te_p, te_t = evaluate(model, loaders['test'], device, criterion)
    te_bin = [1 if x > 0.5 else 0 for x in te_t]
    te_pred = [1 if x > 0.5 else 0 for x in te_p]
    print(f"\n✅ Final Test Acc: {accuracy_score(te_bin, te_pred):.4f}, AUC: {roc_auc_score(te_bin, te_p):.4f}")

if __name__ == '__main__':
    main()
