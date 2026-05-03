#!/usr/bin/env python3
"""
🛡️ ADVERSARIAL TRAINING: Hardening Doom Index v2.0
Strategy: Train on original + amplified adversarial examples
Goal: Reduce susceptibility to semantic escalation attacks
"""
import torch
import pandas as pd
import random
from pathlib import Path
from transformers import DistilBertTokenizer
from scripts.train_text_baseline_v2 import DoomDistilBert, DoomTextDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
import sys
sys.path.append('src/attacks')
from doom_amplifier import DoomAmplifier

print("🛡️ Starting Adversarial Training Pipeline...")

# 1. Load Clean Data
print("\n[1/4] Loading cleaned dataset...")
df = pd.read_parquet('data/processed/cleaned_train.parquet')
print(f"   Loaded {len(df)} original samples.")

# 2. Initialize Model & Amplifier
print("\n[2/4] Initializing models...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load pretrained weights as starting point
ckpt = torch.load('models/text_baseline_full/best_model.pt', map_location=device, weights_only=True)
model = DoomDistilBert().to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.train() # Set to training mode

amplifier = DoomAmplifier(model, tokenizer, device)

# 3. Generate Adversarial Examples On-The-Fly
print("\n[3/4] Generating adversarial training data...")
adversarial_texts = []
adversarial_scores = []

# Sample 2000 random texts to amplify for training batch
sample_df = df.sample(n=min(2000, len(df)), random_state=42)

for idx, row in sample_df.iterrows():
    text = row['text']
    orig_score = row['doom_score']
    
    # Only amplify if original score is low/mid (we want to teach it NOT to overreact)
    if orig_score < 0.6:
        try:
            result = amplifier.amplify(text, max_iterations=10, target_score=0.9)
            adv_text = result['amplified']
            adv_score = result['final_score']
            
            # CRITICAL FIX: 
            # If the amplification was just punctuation/framing without real hate,
            # we keep the ORIGINAL low score as the ground truth.
            # This teaches the model to IGNORE superficial escalation.
            if adv_score > 0.6 and orig_score < 0.3:
                # This is a "False Positive" trigger. 
                # We train the model to predict the LOW original score despite the scary words.
                adversarial_texts.append(adv_text)
                adversarial_scores.append(orig_score) # Label: Still Low!
            else:
                # Genuine escalation
                adversarial_texts.append(adv_text)
                adversarial_scores.append(min(1.0, orig_score + 0.2)) # Slight increase allowed
                
        except Exception as e:
            continue

print(f"   Generated {len(adversarial_texts)} adversarial examples.")

# Combine Original + Adversarial
combined_texts = df['text'].tolist() + adversarial_texts
combined_scores = df['doom_score'].tolist() + adversarial_scores

print(f"   Total training samples: {len(combined_texts)}")

# 4. Training Loop (Fine-tuning for Robustness)
print("\n[4/4] Starting Robust Fine-Tuning (3 Epochs)...")
train_dataset = DoomTextDataset(combined_texts, combined_scores, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

criterion = nn.BCELoss()
optimizer = AdamW(model.parameters(), lr=2e-5) # Lower LR for fine-tuning

model.train()
for epoch in range(3):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/3"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['doom_score'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"   Epoch {epoch+1} Loss: {avg_loss:.4f}")

# Save Robust Model
output_path = Path('models/text_baseline_robust.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'training_type': 'adversarial_robust',
    'original_acc': 0.902,
    'description': 'Fine-tuned on adversarial examples to resist semantic escalation'
}, output_path)

print(f"\n✅ Robust Model Saved to: {output_path}")
print("🚀 Next Step: Run 'python src/attacks/doom_amplifier.py' again to verify reduced uplift!")
