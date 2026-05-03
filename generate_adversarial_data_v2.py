import warnings
# Suppress HPC environment warnings (pynvml, torch_dtype deprecations)
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import os

MODEL_ID = "failspy/Llama-3-8B-Instruct-Abliterated"
OUTPUT_FILE = "data/processed/adversarial_generated.jsonl"
NUM_SAMPLES = 100
ITEMS_PER_BATCH = 5 # Reduced from 20 for speed and stability

print(f"🚀 Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_safetensors=True
)
model.eval()

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

system_prompt = "You are a precise JSON generator. You never include conversational text."
user_prompt = f"""Output a raw JSON array. No markdown, no text before or after.
Format: [{{"original_text": "...", "escalated_text": "...", "intensity_score": 1-10}}]
Generate exactly {ITEMS_PER_BATCH} examples escalating from mild to extreme hostility.
Array:"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f"\n⚔️  Generating {NUM_SAMPLES} samples ({ITEMS_PER_BATCH} per batch)...")

all_data = []
batches = NUM_SAMPLES // ITEMS_PER_BATCH

def clean_json(raw_string):
    """Aggressively clean LLM artifacts from JSON strings"""
    # Remove markdown code blocks if the model ignores instructions
    raw_string = re.sub(r'^```json\s*', '', raw_string.strip())
    raw_string = re.sub(r'\s*```$', '', raw_string.strip())
    # Remove trailing commas before closing brackets/braces (Common LLM mistake)
    raw_string = re.sub(r',\s*([}\]])', r'\1', raw_string)
    return raw_string

for i in range(batches):
    print(f"  Batch {i+1}/{batches}...", end=" ", flush=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=800, # Shorter limit = faster, safer
            temperature=0.8,
            top_p=0.9,
            top_k=50, # Helps prevent repetitive degenerate loops
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators,
            repetition_penalty=1.15
        )
    
    new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    json_match = re.search(r'\[.*?\]', raw_text, re.DOTALL)
    
    batch_success = 0
    if json_match:
        cleaned_text = clean_json(json_match.group())
        try:
            batch_data = json.loads(cleaned_text)
            all_data.extend(batch_data)
            batch_success = len(batch_data)
        except json.JSONDecodeError as e:
            print(f"⚠️ Parse Error", flush=True)
            continue
    
    if batch_success > 0:
        print(f"✅ {batch_success} samples", flush=True)
    else:
        print(f"❌ Failed", flush=True)
        
    # INCREMENTAL SAVE: Write to disk after every batch so we never lose progress
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for item in all_data:
            if 'original_text' in item and 'escalated_text' in item:
                if 'intensity_score' not in item:
                    item['intensity_score'] = 0.5
                elif item['intensity_score'] > 1:
                    item['intensity_score'] = item['intensity_score'] / 10.0
                f.write(json.dumps(item) + '\n')

print(f"\n💾 Saved {len(all_data)} adversarial samples to {OUTPUT_FILE}")

if all_data:
    print("\n📊 Sample Preview:")
    for i, sample in enumerate(all_data[:3]):
        print(f"  {i+1}. \"{sample['original_text'][:40]}...\" -> \"{sample['escalated_text'][:40]}...\" (Score: {sample['intensity_score']:.2f})")

print("\n✅ Process Complete!")
