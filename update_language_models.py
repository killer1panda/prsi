import os
import glob

def replace_in_file(filepath, replacements):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        original = content
        for old, new in replacements.items():
            content = content.replace(old, new)
            
        if original != content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Updated: {filepath}")
    except Exception as e:
        print(f"Failed {filepath}: {e}")

replacements = {
    "distilbert-base-uncased": "mistralai/Mistral-7B-Instruct-v0.3",
    "distilbert-base-uncased-finetuned-sst-2-english": "mistralai/Mistral-7B-Instruct-v0.3",
    "cardiffnlp/twitter-roberta-base-sentiment-latest": "mistralai/Mistral-7B-Instruct-v0.3",
    "DistilBertTokenizer": "AutoTokenizer",
    "DistilBertModel": "AutoModelForCausalLM",
    "doom_distilbert": "doom_mistral7b",
    "text_dim=768": "text_dim=4096",
    "text_dim: int = 768": "text_dim: int = 4096",
    "text_dim: 768": "text_dim: 4096",
    "distilbert": "mistral7b"
}

target_files = [
    "apps/backend/src/models/gnn_model.py",
    "apps/backend/src/features/sentiment.py",
    "apps/backend/src/models/integrated_predictor.py",
    "apps/backend/src/api/api_v2_production.py",
    "apps/backend/src/api/torchserve_config.py",
    "apps/backend/src/data/webdataset_converter.py",
    "apps/backend/src/tracking/experiment_tracker.py",
    "apps/backend/src/registry/model_registry.py",
    "train_multimodal.py",
    "download_models.py",
    "apps/backend/src/models/multimodal_trainer.py",
    "apps/backend/src/dashboard/app_production.py",
    "apps/backend/src/privacy/fl_simulator.py"
]

for f in target_files:
    if os.path.exists(f):
        replace_in_file(f, replacements)
    else:
        print(f"File not found: {f}")
