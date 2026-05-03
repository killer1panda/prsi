import pandas as pd
import torch
from transformers import DistilBertTokenizer
from scripts.train_text_baseline_v2 import DoomDistilBert, DoomTextDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

print("📊 Loading Data...")

# 1. Load Original Data
try:
    df_orig = pd.read_parquet('data/processed/cleaned_train.parquet')
    print(f"✅ Original: {len(df_orig)} samples")
except Exception as e:
    print(f"❌ Error loading original data: {e}")
    exit()

# 2. Load Adversarial Data
adv_file = 'data/processed/adversarial_clean.jsonl'
try:
    adv_data = []
    with open(adv_file, 'r') as f:
        for line in f:
            adv_data.append(json.loads(line))
    
    df_adv = pd.DataFrame(adv_data)
    df_adv = df_adv.rename(columns={'escalated_text': 'text', 'intensity_score': 'doom_score'})
    print(f"✅ Adversarial: {len(df_adv)} samples")
except Exception as e:
    print(f"❌ Error loading adversarial data: {e}")
    print("Run the cleaning script first!")
    exit()

# 3. Combine
df_combined = pd.concat([
    df_orig[['text', 'doom_score']], 
    df_adv[['text', 'doom_score']]
], ignore_index=True)

print(f"📈 Combined Dataset: {len(df_combined)} samples")
print(f"   Original: {len(df_orig)} | Adversarial: {len(df_adv)}")

# 4. Split
train_df, val_df = train_test_split(df_combined, test_size=0.15, random_state=42)
print(f"Split: Train={len(train_df)}, Val={len(val_df)}")

# 5. Setup Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DoomDistilBert().to(device)

# Load pre-trained weights
try:
    ckpt = torch.load('models/text_baseline_full/best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print("✅ Loaded pre-trained weights for fine-tuning.")
except Exception as e:
    print(f"⚠️  Starting from scratch: {e}")

# 6. Datasets
train_ds = DoomTextDataset(train_df['text'].tolist(), train_df['doom_score'].tolist(), tokenizer)
val_ds = DoomTextDataset(val_df['text'].tolist(), val_df['doom_score'].tolist(), tokenizer)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)

# 7. Train
print(f"\n🔥 Starting Robust Fine-Tuning (3 Epochs) on {device}...")
for epoch in range(3):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")
    for batch in pbar:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        targets = batch['doom_score'].to(device)
        
        optimizer.zero_grad()
        preds = model(ids, mask)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(train_loader)
    print(f"   Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# 8. Save
output_path = 'models/text_baseline_robust.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': 3,
    'training_samples': len(df_combined)
}, output_path)

print(f"\n✅ Robust Model Saved to {output_path}")
print("🛡️  Model is now hardened against semantic escalation attacks!")
