import os

filepath = "apps/backend/src/models/gnn_model.py"
try:
    with open(filepath, 'r') as f:
        content = f.read()

    # Apply DistilBERT -> Mistral logic specifically to gnn_model.py since the simple replace missed some formatting
    content = content.replace('DistilBertModel', 'AutoModelForCausalLM')
    content = content.replace('DistilBertTokenizer', 'AutoTokenizer')
    content = content.replace('distilbert-base-uncased', 'mistralai/Mistral-7B-Instruct-v0.3')
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Updated: {filepath}")
except Exception as e:
    print(f"Failed {filepath}: {e}")
