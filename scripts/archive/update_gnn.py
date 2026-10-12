import os

gnn_file = "apps/backend/src/models/gnn_model.py"
with open(gnn_file, 'r') as f:
    content = f.read()
    
# Convert DistilBert/SAGE to Mistral/Hypergraph in the docstrings
content = content.replace("GraphSAGE for user network embeddings", "Hypergraph HGNN + CompGCN for user network embeddings")
content = content.replace("DistilBERT for text", "Mistral-7B QLoRA for text")

with open(gnn_file, 'w') as f:
    f.write(content)
