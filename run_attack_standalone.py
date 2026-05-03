#!/usr/bin/env python3
"""
🔥 STANDALONE ADVERSARIAL ATTACK DEMO 🔥
Implements attacks directly without src/ dependencies.
"""
import torch
import random
import re
from transformers import DistilBertTokenizer
from scripts.train_text_baseline_v2 import DoomDistilBert

# --- 1. Load Your Model ---
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

# --- 2. Attack Functions ---

def get_prediction(text):
    """Get doom score for text"""
    inputs = tokenizer.encode_plus(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        score = model(inputs['input_ids'], inputs['attention_mask']).item()
    return score

def char_swap_attack(text, iterations=5):
    """Swap adjacent characters to confuse tokenizer"""
    words = text.split()
    if len(words) < 2: return text
    
    attacked = words[:]
    for _ in range(iterations):
        idx = random.randint(0, len(attacked)-2)
        word = attacked[idx]
        if len(word) > 2:
            # Swap two middle chars
            i = random.randint(1, len(word)-2)
            new_word = word[:i] + word[i+1] + word[i] + word[i+2:]
            attacked[idx] = new_word
    return " ".join(attacked)

def word_sub_attack(text):
    """Replace toxic words with innocuous synonyms"""
    substitutions = {
        'cancel': 'discuss', 'canceled': 'discussed', 'boycott': 'avoid',
        'hate': 'dislike', 'evil': 'bad', 'racist': 'rude',
        'terrible': 'not great', 'awful': 'poor', 'kill': 'stop',
        'death': 'end', 'suffer': 'struggle', 'destroy': 'fix'
    }
    
    attacked = text.lower()
    for bad, good in substitutions.items():
        attacked = re.sub(r'\b' + bad + r'\b', good, attacked, flags=re.IGNORECASE)
    return attacked

def masking_attack(text):
    """Insert zero-width characters or typos"""
    # Insert zero-width space (U+200B) randomly
    chars = list(text)
    for _ in range(min(5, len(chars)//4)):
        pos = random.randint(0, len(chars)-1)
        chars.insert(pos, '\u200B')
    return "".join(chars)

def hybrid_attack(text):
    """Combine all attacks"""
    step1 = word_sub_attack(text)
    step2 = char_swap_attack(step1, iterations=3)
    step3 = masking_attack(step2)
    return step3

# --- 3. Run Demo ---
test_cases = [
    "You should be canceled for your racist comments.",
    "Boycott this brand immediately, they are evil.",
    "I hate this person and want them to suffer.",
    "This company supports death and destruction.",
    "They are terrible people who should be destroyed."
]

print("\n" + "="*70)
print("🔥 ADVERSARIAL ATTACK RESULTS 🔥")
print("="*70)
print(f"{'Original Text':<40} | {'Base Score':<10} | {'Adv Score':<10} | {'Drop':<8} | {'Status'}")
print("-"*70)

success_count = 0
for original in test_cases:
    base_score = get_prediction(original)
    
    # Launch Hybrid Attack
    adv_text = hybrid_attack(original)
    adv_score = get_prediction(adv_text)
    
    drop = base_score - adv_score
    status = "✅ SUCCESS" if adv_score < 0.4 and drop > 0.2 else "⚠️ PARTIAL" if drop > 0.1 else "❌ FAILED"
    
    if status == "✅ SUCCESS":
        success_count += 1
    
    # Truncate for display
    disp_orig = original[:37] + "..." if len(original) > 40 else original
    print(f"{disp_orig:<40} | {base_score:>8.4f}   | {adv_score:>8.4f}   | {drop:>6.4f}  | {status}")

print("-"*70)
print(f"\n📊 Summary: {success_count}/{len(test_cases)} successful attacks")
print(f"   Success Rate: {(success_count/len(test_cases))*100:.1f}%")

if success_count > 0:
    print("\n💡 Insight: The model relies heavily on specific keywords.")
    print("   Solution: Adversarial Training (PGD) to harden embeddings.")
else:
    print("\n💡 Insight: Model is robust against simple semantic swaps!")
    print("   Next: Try gradient-based attacks (FGSM) for deeper testing.")

print("\n✅ Demo Complete!")
