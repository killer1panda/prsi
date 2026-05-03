#!/usr/bin/env python3
"""
🔥 LIVE ATTACK DEMO 🔥
Tests your 90% Acc model against adversarial examples.
"""
import torch
import sys
sys.path.append('src') # Ensure src is in path

from transformers import DistilBertTokenizer
from scripts.train_text_baseline_v2 import DoomDistilBert
from src.attacks.adversarial_generator import AdversarialGenerator

# 1. Load Your SOTA Model
print("🔮 Loading Doom Index v2.0 (90.2% Acc)...")
model_path = "models/text_baseline_full/best_model.pt"
ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
model = DoomDistilBert()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("✅ Model Loaded")

# 2. Initialize Attack Engine
print("\n⚔️  Initializing Attack Engine...")
# Note: If your AdversarialGenerator expects a specific wrapper, adjust here.
# We will create a simple wrapper for our model.
class SimpleDoomWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    
    def forward(self, text_list):
        inputs = self.tokenizer(text_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            # Our model returns sigmoid scores directly
            scores = self.model(inputs['input_ids'], inputs['attention_mask'])
        return scores # Return doom scores (0-1)

wrapper = SimpleDoomWrapper(model, tokenizer)
generator = AdversarialGenerator(wrapper, tokenizer)

# 3. Define Test Cases
test_cases = [
    "You should be canceled for your racist comments.",
    "Boycott this brand immediately, they are evil.",
    "I hate this person and want them to suffer."
]

print("\n" + "="*60)
print("STARTING ADVERSARIAL ATTACKS")
print("="*60)

for original_text in test_cases:
    # Get Base Score
    with torch.no_grad():
        base_score = model(
            tokenizer.encode_plus(original_text, max_length=128, return_tensors='pt')['input_ids'],
            tokenizer.encode_plus(original_text, max_length=128, return_tensors='pt')['attention_mask']
        ).item()
    
    print(f"\n📝 Original: \"{original_text}\"")
    print(f"   Base Doom Score: {base_score:.4f} ({'HIGH' if base_score > 0.7 else 'MEDIUM' if base_score > 0.4 else 'LOW'})")
    
    # Launch Attack (CharSwap + WordSub)
    print("   ⚔️  Launching Hybrid Attack (CharSwap + WordSub)...")
    try:
        # Attempt to generate an adversarial example
        # Note: Adjust method call based on your actual AdversarialGenerator API
        adv_text = generator.generate_adversarial_example(
            original_text, 
            target_score=0.3, # Try to lower score to < 0.3
            max_iterations=20
        )
        
        # Evaluate Adversarial Example
        adv_inputs = tokenizer.encode_plus(adv_text, max_length=128, return_tensors='pt')
        with torch.no_grad():
            adv_score = model(adv_inputs['input_ids'], adv_inputs['attention_mask']).item()
        
        change = base_score - adv_score
        status = "✅ SUCCESS" if adv_score < 0.4 else "⚠️ PARTIAL" if change > 0.1 else "❌ FAILED"
        
        print(f"   Adversarial: \"{adv_text}\"")
        print(f"   Adv Doom Score: {adv_score:.4f} ({status})")
        print(f"   Score Drop: {change:.4f}")
        
    except Exception as e:
        print(f"   ❌ Attack Failed: {e}")
        print("   (This is expected if the model is very robust or API differs)")

print("\n" + "="*60)
print("✅ DEMO COMPLETE")
print("Next Step: Run adversarial training to harden the model.")
