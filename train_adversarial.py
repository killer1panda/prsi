#!/usr/bin/env python3
"""
🛡️ ADVERSARIAL TRAINING SCRIPT
Hardens the model against punctuation and framing attacks.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from scripts.train_text_baseline_v2 import DoomDistilBert
from src.attacks.adversarial_generator import AdversarialGenerator, AttackResult
import random

# --- Configuration ---
MODEL_PATH = "models/text_baseline_full/best_model.pt"
OUTPUT_PATH = "models/text_baseline_robust.pt"
EPOCHS = 3
ADV_RATIO = 0.5  # 50% of batches will be adversarial

# --- Load Model ---
print("🛡️ Loading Model for Hardening...")
ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
model = DoomDistilBert()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# --- Simple Predictor Wrapper ---
class SimplePredictor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def predict(self, text, author_id="anon"):
        inputs = self.tokenizer.encode_plus(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            score = self.model(inputs['input_ids'], inputs['attention_mask']).item()
        return {'probability': score, 'label': 1 if score > 0.5 else 0}

predictor = SimplePredictor(model, tokenizer, device)
generator = AdversarialGenerator(predictor, max_iterations=10, population_size=5)

# --- Adversarial Dataset ---
# Load your original training data
import pandas as pd
from pathlib import Path
train_df = pd.read_parquet(Path("data/processed/cleaned_train.parquet"))
texts = train_df['text'].tolist()
scores = train_df['doom_score'].tolist()

print(f"📚 Original Training Samples: {len(texts)}")
print(f"⚔️  Generating Adversarial Examples...")

adv_texts = []
adv_scores = []

# Generate adversarial examples for a subset
sample_size = min(1000, len(texts))
indices = random.sample(range(len(texts)), sample_size)

for i in indices:
    orig_text = texts[i]
    orig_score = scores[i]
    
    try:
        # Generate attacks
        results = generator.generate_variants(orig_text, max_variants=2, use_genetic=False)
        
        # Add successful attacks to training set
        for res in results:
            if res.doom_uplift > 0.05: # Only keep if it changed the score significantly
                adv_texts.append(res.variant_text)
                # Use the original score as ground truth (we want model to ignore the attack)
                adv_scores.append(orig_score)
    except Exception as e:
        continue

print(f"✅ Generated {len(adv_texts)} adversarial examples.")

# Combine datasets
all_texts = texts + adv_texts
all_scores = scores + adv_scores

print(f"📈 New Total Samples: {len(all_texts)}")

# --- Fine-tuning Loop ---
class SimpleDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_len=128):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(self.texts[idx], max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'label': torch.tensor(self.scores[idx], dtype=torch.float32)
        }

dataset = SimpleDataset(all_texts, all_scores, tokenizer)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print(f"\n🏋️ Starting Adversarial Training ({EPOCHS} epochs)...")

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        targets = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(ids, mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# Save Robust Model
torch.save({
    'model_state_dict': model.state_dict(),
    'training_type': 'adversarial',
    'epochs': EPOCHS
}, OUTPUT_PATH)

print(f"\n✅ Robust Model Saved to {OUTPUT_PATH}")
print("💡 Next: Re-run the attack demo to verify improved robustness!")
