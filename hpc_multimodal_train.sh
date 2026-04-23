#!/bin/bash
#PBS -N doom_multimodal_train
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:ncpus=64:ngpus=4:mem=512gb
#PBS -l walltime=24:00:00
#PBS -o logs/doom_multimodal_train.log

cd $PBS_O_WORKDIR
mkdir -p logs

echo "=========================================="
echo "Doom Index Multimodal Training"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# ── Environment ─────────────────────────────────────────────────────────────

# Load modules (adjust for your HPC)
module purge
module load cuda/12.2
module load cudnn/8.9
module load anaconda3/2023.09

# Activate conda env
conda activate doom || {
    echo "Creating conda environment..."
    conda create -n doom python=3.10 -y
    conda activate doom
}

# ── Install Dependencies ────────────────────────────────────────────────────

echo "Installing dependencies..."

# PyTorch for CUDA 12.1 (compatible with H100 sm_90)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric (for GraphSAGE)
pip install torch-geometric==2.4.0
pip install pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Transformers & NLP
pip install transformers==4.36.0 datasets==2.15.0 accelerate==0.25.0

# ML & Data
pip install scikit-learn==1.3.2 pandas==2.1.4 numpy==1.26.3

# API & Utils
pip install fastapi==0.105.0 uvicorn==0.24.0 pydantic==2.5.0
pip install loguru==0.7.2 tqdm==4.66.1

# Neo4j & MongoDB
pip install neo4j==5.15.0 pymongo==4.6.1

# Optional: WandB for experiment tracking
# pip install wandb

echo "Dependencies installed."

# ── Data Preparation ────────────────────────────────────────────────────────

echo "Preparing data..."

# If Pushshift data exists, process it
if [ -d "doom\ index/data/twitter_dataset/scraped_data/reddit" ]; then
    echo "Found Reddit data. Processing..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from train_model_full import process_ndjson_to_df, add_engineered_features, create_feature_matrix
import pandas as pd
import glob

# Process all available NDJSON files
all_dfs = []
for f in glob.glob('doom\ index/data/twitter_dataset/scraped_data/reddit/comments/*.ndjson')[:6]:
    try:
        df = process_ndjson_to_df(f, is_comment=True)
        if len(df) > 0:
            all_dfs.append(df)
            print(f'Processed {f}: {len(df)} rows')
    except Exception as e:
        print(f'Failed {f}: {e}')

for f in glob.glob('doom\ index/data/twitter_dataset/scraped_data/reddit/submissions/*.ndjson')[:6]:
    try:
        df = process_ndjson_to_df(f, is_comment=False)
        if len(df) > 0:
            all_dfs.append(df)
            print(f'Processed {f}: {len(df)} rows')
    except Exception as e:
        print(f'Failed {f}: {e}')

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = add_engineered_features(combined)

    # Create PROPER weak labels (not just keyword presence)
    def label_cancellation_event(row):
        score = 0
        if row['likes'] > 100: score += 1
        if row.get('sentiment_polarity', 0) < -0.3: score += 1
        text_lower = str(row.get('text', '')).lower()
        if any(kw in text_lower for kw in ['boycott', 'petition', 'fired', 'apologized', 'removed', 'banned']): score += 2
        if any(kw in text_lower for kw in ['cancel', 'backlash', 'outrage', 'controversy']): score += 1
        return 1 if score >= 3 else 0

    combined['label'] = combined.apply(label_cancellation_event, axis=1)

    # Balance dataset
    pos = combined[combined['label'] == 1]
    neg = combined[combined['label'] == 0].sample(n=min(len(pos)*3, len(neg)), random_state=42)
    balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    balanced.to_csv('data/processed_reddit_multimodal.csv', index=False)
    print(f'Saved balanced dataset: {len(balanced)} rows')
    print(f'Label distribution: {balanced["label"].value_counts().to_dict()}')
else:
    print('No Reddit data found. Using existing CSV.')
"
fi

# ── Training ────────────────────────────────────────────────────────────────

echo "Starting multimodal training with DDP on 4x H100..."

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# Use torchrun for DDP across 4 GPUs
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=doom_train_$$ \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    train_multimodal.py \
    --data_path data/processed_reddit_multimodal.csv \
    --output_dir models/multimodal_doom \
    --epochs 15 \
    --batch_size 16 \
    --lr 2e-5 \
    --graph_hidden 128 \
    --graph_layers 2 \
    --fusion_hidden 256 \
    --freeze_bert_layers 5 \
    --fp16 \
    --ddp \
    --grad_accum 4 \
    --max_length 256

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="

# ── Export to ONNX for fast inference ──────────────────────────────────────

echo "Exporting to ONNX..."
python3 -c "
import torch
from src.models.gnn_model import MultimodalDoomPredictor

# Load best model
config = torch.load('models/multimodal_doom/model_config.pt')
model = MultimodalDoomPredictor(**config)
checkpoint = torch.load('models/multimodal_doom/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().cuda()

# Export text encoder to ONNX (graph part stays in PyG)
dummy_input_ids = torch.randint(0, 30000, (1, 256), dtype=torch.long).cuda()
dummy_attention = torch.ones(1, 256, dtype=torch.long).cuda()

torch.onnx.export(
    model.text_encoder.bert,
    (dummy_input_ids, dummy_attention),
    'models/multimodal_doom/text_encoder.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'last_hidden_state': {0: 'batch_size', 1: 'sequence'},
    },
    opset_version=14,
)
print('ONNX export complete.')
" 2>/dev/null || echo "ONNX export skipped (optional)"

echo "All done. Check models/multimodal_doom/ for outputs."
