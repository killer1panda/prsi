import os

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

vision_replacements = {
    "openai/clip-vit-base-patch32": "Qwen/Qwen2-VL-7B-Instruct",
    "CLIPProcessor": "Qwen2VLForConditionalGeneration, AutoProcessor",
    "CLIPVisionModel": "Qwen2VLForConditionalGeneration",
    "CLIPModel": "Qwen2VLForConditionalGeneration",
    "embedding_dim: int = 512": "embedding_dim: int = 3584",
    "vision_dim: int = 256": "vision_dim: int = 512"
}

vision_files = [
    "apps/backend/src/models/vision_encoder.py",
    "apps/backend/src/models/meme_detector.py",
    "apps/backend/src/models/frontier_multimodal.py",
    "apps/backend/src/models/contrastive_pretrain.py",
    "apps/backend/src/data/qdrant_vector_store.py"
]

for f in vision_files:
    if os.path.exists(f):
        replace_in_file(f, vision_replacements)
