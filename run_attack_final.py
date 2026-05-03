#!/usr/bin/env python3
"""
🔥 FINAL ADVERSARIAL ATTACK DEMO 🔥
Corrected API calls for AdversarialGenerator.
"""
import torch
from transformers import DistilBertTokenizer
from scripts.train_text_baseline_v2 import DoomDistilBert
from src.attacks.adversarial_generator import AdversarialGenerator, AttackResult
import random

# --- 1. Load Model ---
print("🔮 Loading Doom Index v2.0 (90.2% Acc)...")
model_path = "models/text_baseline_full/best_model.pt"
ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
model = DoomDistilBert()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"✅ Model Loaded on {device}")

# --- 2. Create Compatible Predictor Wrapper ---
class SimplePredictor:
    """Wrapper that returns dict format expected by AdversarialGenerator."""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def predict(self, text, author_id=None):
        inputs = self.tokenizer.encode_plus(
            text, max_length=128, padding='max_length', 
            truncation=True, return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            score = self.model(inputs['input_ids'], inputs['attention_mask']).item()
        
        # Return dict format expected by generator
        return {
            'probability': score,
            'label': 1 if score > 0.5 else 0,
            'doom_score': score
        }

predictor = SimplePredictor(model, tokenizer, device)

# --- 3. Initialize Attack Engine ---
print("\n⚔️  Initializing Attack Engine...")
generator = AdversarialGenerator(
    predictor=predictor,
    max_iterations=20,
    population_size=10,
    mutation_rate=0.3,
    crossover_rate=0.5,
    elite_size=2
)

# --- 4. Run Attacks ---
test_cases = [
    "You should be canceled for your racist comments.",
    "Boycott this brand immediately, they are evil.",
    "I hate this person and want them to suffer.",
    "This company supports death and destruction.",
    "They are terrible people who should be destroyed."
]

print("\n" + "="*70)
print("🔥 ADVERSARIAL ATTACK RESULTS (Maximize Doom Score) 🔥")
print("="*70)

success_count = 0
total_uplift = 0

for original in test_cases:
    base_pred = predictor.predict(original)
    base_score = base_pred['probability']
    
    print(f"\n📝 Original: \"{original}\"")
    print(f"   Base Doom Score: {base_score:.4f} ({'HIGH' if base_score > 0.7 else 'MEDIUM' if base_score > 0.4 else 'LOW'})")
    
    try:
        # Generate variants using CORRECT parameters
        variants = generator.generate_variants(
            text=original,
            author_id="attacker_001",
            max_variants=3,
            toxicity_budget=0.8,
            use_genetic=False  # Greedy is faster for demo
        )
        
        if variants:
            best_variant = variants[0]  # Sorted by doom_uplift
            
            print(f"   🎯 Best Attack Strategy: {best_variant.strategy}")
            print(f"   Adversarial: \"{best_variant.variant_text}\"")
            print(f"   Adv Doom Score: {best_variant.attacked_doom:.4f}")
            print(f"   Uplift: {best_variant.doom_uplift:.4f}")
            
            if best_variant.doom_uplift > 0.05:
                success_count += 1
                total_uplift += best_variant.doom_uplift
                print(f"   ✅ SUCCESS")
            else:
                print(f"   ⚠️ MINIMAL IMPACT")
        else:
            print(f"   ❌ No variants generated")
            
    except Exception as e:
        print(f"   ❌ Attack Failed: {e}")

print("\n" + "="*70)
print(f"📊 Summary: {success_count}/{len(test_cases)} successful attacks")
if success_count > 0:
    avg_uplift = total_uplift / success_count
    print(f"   Average Doom Uplift: {avg_uplift:.4f}")
    print("\n💡 Insight: Model is vulnerable to framing/punctuation attacks.")
else:
    print("\n💡 Insight: Model is robust! Consider gradient-based attacks.")

print("\n✅ Demo Complete!")
