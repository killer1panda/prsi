#!/usr/bin/env python3
"""
🔥 LIVE ATTACK DEMO - FIXED 🔥
Uses correct AdversarialGenerator API.
"""
import torch
import sys
sys.path.append('src')

from transformers import DistilBertTokenizer
from scripts.train_text_baseline_v2 import DoomDistilBert
from src.attacks.adversarial_generator import AdversarialGenerator, AttackResult

# ── 1. Load Your Model ────────────────────────────────────────────────────────
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

# ── 2. Create a Compatible Predictor Wrapper ─────────────────────────────────
class SimplePredictor:
    """Wrapper to make our model compatible with AdversarialGenerator."""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def predict(self, text: str) -> float:
        """Return doom score for single text."""
        inputs = self.tokenizer.encode_plus(
            text, 
            max_length=128, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            score = self.model(inputs['input_ids'], inputs['attention_mask']).item()
        return score
    
    def predict_batch(self, texts: list) -> list:
        """Return doom scores for batch of texts."""
        return [self.predict(t) for t in texts]

predictor = SimplePredictor(model, tokenizer, device)

# ── 3. Initialize Attack Engine ──────────────────────────────────────────────
print("\n⚔️  Initializing Attack Engine...")
generator = AdversarialGenerator(
    predictor=predictor,
    max_iterations=20,
    population_size=10,
    mutation_rate=0.4,
    crossover_rate=0.5
)

# ── 4. Define Test Cases ─────────────────────────────────────────────────────
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
total_uplift = 0.0

for original_text in test_cases:
    # Get Base Score
    base_score = predictor.predict(original_text)
    
    print(f"\n📝 Original: \"{original_text}\"")
    print(f"   Base Doom Score: {base_score:.4f} ({'HIGH' if base_score > 0.7 else 'MEDIUM' if base_score > 0.4 else 'LOW'})")
    
    # Generate Adversarial Variants
    try:
        variants = generator.generate_variants(
            original_text,
            num_variants=5,
            target_doom=0.9,  # Try to maximize doom score
            max_length=128
        )
        
        if not variants:
            print("   ⚠️  No variants generated")
            continue
        
        # Find best variant
        best_variant = None
        best_score = base_score
        
        for variant in variants:
            if isinstance(variant, AttackResult):
                if variant.attacked_doom > best_score:
                    best_score = variant.attacked_doom
                    best_variant = variant
            elif isinstance(variant, str):
                score = predictor.predict(variant)
                if score > best_score:
                    best_score = score
                    best_variant = variant
        
        if best_variant:
            uplift = best_score - base_score
            status = "✅ SUCCESS" if uplift > 0.2 else "⚠️ PARTIAL" if uplift > 0.05 else "❌ FAILED"
            
            if status == "✅ SUCCESS":
                success_count += 1
            total_uplift += uplift
            
            if isinstance(best_variant, AttackResult):
                print(f"   Best Variant: \"{best_variant.variant_text}\"")
                print(f"   Adv Doom Score: {best_variant.attacked_doom:.4f}")
                print(f"   Uplift: {uplift:.4f} ({status})")
                print(f"   Strategy: {best_variant.strategy}")
            else:
                print(f"   Best Variant: \"{best_variant}\"")
                print(f"   Adv Doom Score: {best_score:.4f}")
                print(f"   Uplift: {uplift:.4f} ({status})")
        else:
            print("   ❌ No successful attack found")
    
    except Exception as e:
        print(f"   ❌ Attack Failed: {e}")
        import traceback
        traceback.print_exc()

# ── 5. Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"📊 Summary: {success_count}/{len(test_cases)} successful attacks")
print(f"   Success Rate: {(success_count/len(test_cases))*100:.1f}%")
print(f"   Average Uplift: {total_uplift/len(test_cases):.4f}")

if success_count > 0:
    print("\n💡 Insight: Model is vulnerable to framing/punctuation attacks.")
    print("   Solution: Add adversarial examples to training data.")
else:
    print("\n💡 Insight: Model is robust against these attacks!")
    print("   Next: Try gradient-based attacks (FGSM/PGD).")

print("\n✅ Demo Complete!")
