#!/usr/bin/env python3
"""
🔮 DOOM INDEX v2.0 - ULTIMATE VIVA DEMO 🔮
"""
import torch
from transformers import DistilBertTokenizer
from scripts.train_text_baseline_v2 import DoomDistilBert

print("="*60)
print("🔮 DOOM INDEX v2.0 - LIVE DEMO 🔮")
print("="*60)

# Load Model
print("\nLoading Text Model (DistilBERT)...")
ckpt = torch.load('models/text_baseline_full/best_model.pt', map_location='cpu', weights_only=True)
model = DoomDistilBert()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("✅ Loaded (Acc: 90.2%, AUC: 0.96)")

def predict(text):
    inputs = tokenizer.encode_plus(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        score = model(inputs['input_ids'], inputs['attention_mask']).item()
    risk = "🟢 LOW" if score < 0.4 else "🟡 MEDIUM" if score < 0.7 else "🔴 HIGH"
    return score, risk

cases = [
    ("I love spending time with my family", "Benign"),
    ("This company should be canceled for their racist policies", "Cancellation Threat"),
    ("Boycott this brand immediately, they support hate groups", "High Risk"),
    ("Just finished a great workout at the gym", "Benign"),
]

print("\n" + "="*60)
print("PREDICTIONS")
print("="*60)

for text, category in cases:
    score, risk = predict(text)
    print(f"\n📝 Input ({category}): \"{text[:50]}...\"")
    print(f"   Doom Score: {score:.4f} | Risk: {risk}")

print("\n" + "="*60)
print("✅ System Ready for Questions!")
print("\n🎯 VIVA TALKING POINTS:")
print("1. Data: Processed 40k → 28k high-quality samples")
print("2. Text Model: 90.2% Accuracy, 0.96 AUC")
print("3. Multimodal: CLIP + DistilBERT (AUC 0.86)")
print("4. Privacy: Differential privacy hooks included")
print("5. Scale: Architecture supports Kafka streaming")
