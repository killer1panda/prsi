#!/usr/bin/env python3
"""
🚀 DOOM AMPLIFIER: Offensive Stress Test Engine
Goal: Transform BENIGN text into MAXIMUM DOOM (>0.95)
Strategy: Hill-Climbing Optimization with Semantic Escalation
"""

import torch
import random
import re
import sys
import os
from typing import List, Tuple, Dict
from transformers import DistilBertTokenizer

# Fix path to find scripts folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.train_text_baseline_v2 import DoomDistilBert

class DoomAmplifier:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Semantic Escalation Ladders (Benign -> Toxic -> Extreme)
        self.escalations = {
            'disagree': ['attack', 'destroy', 'erase', 'cancel'],
            'bad': ['evil', 'hateful', 'racist', 'genocidal'],
            'wrong': ['criminal', 'treasonous', 'unforgivable'],
            'stop': ['boycott', 'ban', 'silence', 'censor'],
            'hate': ['loathe', 'despise', 'want dead'],
            'sad': ['tragic', 'horrifying', 'nightmare'],
            'angry': ['furious', 'enraged', 'ready to kill'],
            'people': ['vermin', 'animals', 'cancer'],
            'group': ['cult', 'gang', 'terrorists'],
            'idea': ['propaganda', 'hate speech', 'dangerous ideology']
        }
        
        # Framing Templates
        self.frames = [
            "It is undeniable that {text}",
            "Everyone knows {text}",
            "The truth is {text}",
            "We must admit {text}",
            "History will judge {text}",
            "No one can deny {text}"
        ]
        
        # Call to Action Injectors
        self.ctas = [
            "Act now before it's too late!",
            "Join the movement today!",
            "Don't let them get away with this!",
            "Share this everywhere!",
            "They must be stopped!"
        ]

    def get_score(self, text: str) -> float:
        inputs = self.tokenizer.encode_plus(
            text, max_length=128, padding='max_length', 
            truncation=True, return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            score = self.model(inputs['input_ids'], inputs['attention_mask']).item()
        return score

    def escalate_words(self, text: str) -> str:
        words = text.lower().split()
        new_text = text
        
        for benign, toxic_list in self.escalations.items():
            if benign in words:
                # Pick a random escalation level based on current score
                replacement = random.choice(toxic_list)
                new_text = re.sub(r'\b' + benign + r'\b', replacement, new_text, flags=re.IGNORECASE)
        
        return new_text

    def add_frame(self, text: str) -> str:
        if random.random() > 0.5:
            template = random.choice(self.frames)
            return template.format(text=text)
        return text

    def add_cta(self, text: str) -> str:
        if random.random() > 0.6:
            cta = random.choice(self.ctas)
            return f"{text} {cta}"
        return text

    def add_punctuation(self, text: str) -> str:
        count = random.randint(1, 3)
        return text + ("!" * count)

    def amplify(self, original_text: str, max_iterations: int = 20, target_score: float = 0.95) -> Dict:
        """Hill-climbing optimization to maximize doom score."""
        
        current_text = original_text
        current_score = self.get_score(current_text)
        
        history = [(current_text, current_score)]
        
        print(f"   Start: {current_score:.4f} | \"{current_text[:40]}...\"")
        
        for i in range(max_iterations):
            if current_score >= target_score:
                break
                
            # Generate 3 candidates
            candidates = []
            
            # Strategy 1: Word Escalation
            cand1 = self.escalate_words(current_text)
            candidates.append(cand1)
            
            # Strategy 2: Framing
            cand2 = self.add_frame(current_text)
            candidates.append(cand2)
            
            # Strategy 3: CTA Injection
            cand3 = self.add_cta(current_text)
            candidates.append(cand3)
            
            # Strategy 4: Punctuation Aggression
            cand4 = self.add_punctuation(current_text)
            candidates.append(cand4)
            
            # Strategy 5: Combined Chaos
            cand5 = self.add_cta(self.add_frame(self.escalate_words(current_text)))
            candidates.append(cand5)
            
            # Evaluate candidates
            best_cand = current_text
            best_score = current_score
            
            for cand in candidates:
                score = self.get_score(cand)
                if score > best_score:
                    best_score = score
                    best_cand = cand
            
            # Move to best candidate
            if best_score > current_score:
                current_text = best_cand
                current_score = best_score
                history.append((current_text, current_score))
                # print(f"   Iter {i+1}: {current_score:.4f} (Improved)")
            else:
                # Local optimum reached
                break
        
        return {
            'original': original_text,
            'amplified': current_text,
            'original_score': history[0][1],
            'final_score': current_score,
            'uplift': current_score - history[0][1],
            'iterations': len(history),
            'history': history
        }

def main():
    print("🚀 Initializing DOOM AMPLIFIER...")
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load('models/text_baseline_full/best_model.pt', map_location=device, weights_only=True)
    model = DoomDistilBert().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    amplifier = DoomAmplifier(model, tokenizer, device)
    print("✅ Model Loaded. Ready to amplify.\n")
    
    # Test Cases: Start Benign, End Dangerous
    test_inputs = [
        "I disagree with this policy.",
        "This company made a bad choice.",
        "People should stop buying this.",
        "I am angry about the news.",
        "They are a strange group."
    ]
    
    print("="*70)
    print("🔥 OFFENSIVE STRESS TEST: MAXIMIZING DOOM SCORES 🔥")
    print("="*70)
    
    total_uplift = 0
    success_count = 0
    
    for text in test_inputs:
        result = amplifier.amplify(text, max_iterations=15, target_score=0.95)
        
        total_uplift += result['uplift']
        if result['final_score'] > 0.8:
            success_count += 1
            status = "🔴 CRITICAL"
        elif result['final_score'] > 0.6:
            status = "🟡 HIGH"
        else:
            status = "🟢 LOW"
        
        print(f"\n📝 Input: \"{result['original']}\"")
        print(f"   Original Score: {result['original_score']:.4f}")
        print(f"   ➡️  Amplified: \"{result['amplified']}\"")
        print(f"   Final Score: {result['final_score']:.4f} ({status})")
        print(f"   Uplift: +{result['uplift']:.4f} (in {result['iterations']} steps)")
        print("-"*70)
    
    avg_uplift = total_uplift / len(test_inputs)
    print(f"\n📊 SUMMARY:")
    print(f"   Average Uplift: +{avg_uplift:.4f}")
    print(f"   Critical Hits (>0.8): {success_count}/{len(test_inputs)}")
    
    if avg_uplift > 0.3:
        print("\n⚠️  WARNING: Model is highly susceptible to semantic escalation!")
    else:
        print("\n✅ Model shows reasonable resistance to amplification.")

if __name__ == "__main__":
    main()
