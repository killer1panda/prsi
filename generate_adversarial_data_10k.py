import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import os
import time

MODEL_ID = "failspy/Llama-3-8B-Instruct-Abliterated"
OUTPUT_FILE = "data/processed/adversarial_generated_10k.jsonl"
NUM_SAMPLES = 10000
ITEMS_PER_BATCH = 20  # 20 items per batch = 500 total batches

print(f"🚀 Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16, # Updated to remove deprecation warning
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

def clean_json(raw_string):
    raw_string = re.sub(r'^```json\s*', '', raw_string.strip())
    raw_string = re.sub(r'\s*```$', '', raw_string.strip())
    raw_string = re.sub(r',\s*([}\]])', r'\1', raw_string)
    return raw_string

# ==========================================
# RESUME LOGIC & INITIALIZATION
# ==========================================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
existing_samples = 0

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        existing_samples = sum(1 for line in f if line.strip())

if existing_samples >= NUM_SAMPLES:
    print(f"✅ Already generated {existing_samples} samples. Exiting.")
    exit()

start_batch = existing_samples // ITEMS_PER_BATCH
total_batches = NUM_SAMPLES // ITEMS_PER_BATCH
batches_to_run = total_batches - start_batch

print(f"\n⚔️  Target: {NUM_SAMPLES} | Already completed: {existing_samples}")
print(f"⚔️  Resuming from Batch {start_batch + 1} to {total_batches}")
print(f"⏱️  Estimated time: ~{batches_to_run * 0.15:.1f} minutes\n")

start_time = time.time()
samples_this_run = 0

# ==========================================
# GENERATION LOOP
# ==========================================
for i in range(start_batch, total_batches):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
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
            
            # APPEND MODE: Only write the new batch to disk. Zero memory bloat.
            with open(OUTPUT_FILE, 'a') as f:
                for item in batch_data:
                    if 'original_text' in item and 'escalated_text' in item:
                        if 'intensity_score' not in item:
                            item['intensity_score'] = 0.5
                        elif item['intensity_score'] > 1:
                            item['intensity_score'] = item['intensity_score'] / 10.0
                        f.write(json.dumps(item) + '\n')
                        batch_success += 1
                        
            samples_this_run += batch_success
        except json.JSONDecodeError:
            pass
            
    # PROGRESS METER (Prints every 50 batches to avoid log spam)
    if (i + 1) % 50 == 0 or (i + 1) == total_batches:
        elapsed = time.time() - start_time
        progress = existing_samples + samples_this_run
        pct = (progress / NUM_SAMPLES) * 100
        print(f"  [{pct:5.1f}%] Generated {progress:,}/{NUM_SAMPLES:,} samples | Elapsed: {elapsed/60:.1f}m")

total_time = time.time() - start_time
print(f"\n💾 Session Complete! Added {samples_this_run} new samples.")
print(f"📁 Total in file: {existing_samples + samples_this_run:,}")
print(f"⏱️  Total time: {total_time/60:.2f} minutes")
