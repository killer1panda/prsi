import torch
import pandas as pd
from scripts.train_text_baseline import DoomDistilBert, DoomTextDataset
from transformers import DistilBertTokenizer

checkpoint = torch.load('models/text_baseline/best_model.pt', map_location='cpu')
model = DoomDistilBert()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def predict_doom(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        output = model(inputs['input_ids'], inputs['attention_mask'])
    
    score = output.item()
    risk = "LOW" if score < 0.4 else "MEDIUM" if score < 0.7 else "HIGH"
    return score, risk

examples = [
    "I love cats!",
    "You are terrible and should be canceled",
    "This brand is racist and I will boycott them forever",
    "Just had a great day at work"
]

print("🔮 DOOM INDEX PREDICTIONS 🔮")
print("="*50)
for text in examples:
    score, risk = predict_doom(text)
    print(f"Text: {text[:40]}...")
    print(f"  Doom Score: {score:.4f} | Risk: {risk}")
    print()
