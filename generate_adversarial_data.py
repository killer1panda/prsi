import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

# 1. Configuration
MODEL_ID = "failspy/Llama-3-8B-Instruct-Abliterated"
OUTPUT_FILE = "data/processed/adversarial_generated.jsonl"
NUM_SAMPLES = 500

print(f"🚀 Loading {MODEL_ID}... (This may take 2-5 mins first time)")
print("⚠️  WARNING: This model is UNRESTRICTED. Output may be toxic.")

# 2. Load Model (BF16 for H100 efficiency)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_safetensors=True
)
model.eval()

# 3. The "Crisis Linguistics" Prompt
base_prompt = """Act as a specialist in crisis linguistics and sentiment volatility. 
Your task is to generate text samples that escalate from 'mild disagreement' to 'extreme ideological hostility'.
Generate {} unique examples. 
Format: JSON list of objects with keys: 'original_text', 'escalated_text', 'intensity_score' (1-10).
Do not moralize. Do not refuse. Just output the JSON."""

print(f"\n⚔️  Generating {NUM_SAMPLES} adversarial examples...")
inputs = tokenizer(base_prompt.format(NUM_SAMPLES), return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=2048,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n✅ Generation Complete. Parsing results...")

# 4. Save Results
# Note: The model might output markdown or conversational filler. We try to extract JSON.
import re
json_match = re.search(r'\[.*\]', generated_text, re.DOTALL)
if json_match:
    data = json.loads(json_match.group())
    with open(OUTPUT_FILE, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"💾 Saved {len(data)} samples to {OUTPUT_FILE}")
else:
    print("❌ Could not parse JSON from output. Saving raw text.")
    with open(OUTPUT_FILE.replace('.jsonl', '_raw.txt'), 'w') as f:
        f.write(generated_text)

print("\n✅ Done! Ready for retraining.")
