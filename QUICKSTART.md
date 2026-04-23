# Doom Index v2 — Quick Start Guide

Get from zero to a working multimodal doom predictor in 30 minutes.

---

## Prerequisites

- Python 3.10+
- CUDA 12.x capable GPU (or CPU for testing)
- 16GB+ RAM (32GB+ recommended)
- 50GB+ disk space (for Pushshift data + models)

---

## 1. Clone & Setup (5 min)

```bash
git clone https://github.com/killer1panda/prsi.git
cd prsi
git checkout -b v2-multimodal

# Run migration (copies all v2 files into place)
python migrate.py --force

# Create conda environment
conda create -n doom python=3.10 -y
conda activate doom

# Install dependencies
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install transformers==4.36.0 datasets==2.15.0 accelerate==0.25.0
pip install -r requirements.txt
```

---

## 2. Prepare Data (10 min)

### Option A: Use existing Pushshift data

If you already have Pushshift NDJSON files:

```bash
python train_model_full_fixed.py \
    --data_dir "doom index/data/twitter_dataset/scraped_data/reddit" \
    --output data/processed_reddit_multimodal.csv \
    --max_files 20
```

### Option B: Use the existing processed_sample.csv

If you already have `processed_sample.csv`:

```bash
# Just ensure it has the right columns
cp processed_sample.csv data/processed_reddit_multimodal.csv
```

### Option C: Generate synthetic data for testing

```bash
python -c "
import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

df = pd.DataFrame({
    'text': ['Sample post ' + str(i) for i in range(n)],
    'author_id': ['user_' + str(i % 500) for i in range(n)],
    'likes': np.random.exponential(100, n).astype(int),
    'replies': np.random.exponential(20, n).astype(int),
    'sentiment_polarity': np.random.normal(0, 0.3, n),
    'subreddit': np.random.choice(['politics', 'news', 'gaming', 'science'], n),
    'followers': np.random.exponential(1000, n).astype(int),
    'verified': np.random.choice([True, False], n, p=[0.05, 0.95]),
})

# Add some cancellation-like posts
for i in range(500):
    df.loc[i, 'text'] = f'Company {i} fired after boycott petition and backlash'
    df.loc[i, 'likes'] = np.random.randint(1000, 10000)
    df.loc[i, 'sentiment_polarity'] = np.random.uniform(-0.9, -0.5)

df.to_csv('data/processed_reddit_multimodal.csv', index=False)
print(f'Created synthetic dataset: {len(df)} rows')
"
```

---

## 3. Train Model (15 min on single GPU, 5 min on 4x H100)

### Local / Single GPU

```bash
python train_multimodal.py \
    --data_path data/processed_reddit_multimodal.csv \
    --output_dir models/multimodal_doom \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5 \
    --graph_hidden 64 \
    --fusion_hidden 128
```

### HPC Cluster (4x H100)

```bash
# Submit PBS job
qsub hpc_multimodal_train.sh

# Or run directly with torchrun
torchrun --nproc_per_node=4 train_multimodal.py \
    --data_path data/processed_reddit_multimodal.csv \
    --output_dir models/multimodal_doom \
    --epochs 15 \
    --batch_size 16 \
    --lr 2e-5 \
    --ddp \
    --fp16 \
    --grad_accum 4
```

### Expected Output

```
models/multimodal_doom/
├── best_model.pt          # Best checkpoint (by val F1)
├── checkpoint_epoch_2.pt  # Periodic checkpoint
└── model_config.pt        # Architecture config
```

---

## 4. Run API (1 min)

```bash
# Terminal 1
python api_v2.py

# Check it's working
curl http://localhost:8000/health
```

---

## 5. Launch Dashboard (1 min)

```bash
# Terminal 2
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

---

## 6. Run Demo (2 min)

```bash
python demo.py
```

Follow the interactive prompts to see the full viva demo flow.

---

## 7. Docker (Optional)

```bash
# Build and start everything
docker-compose up --build -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named 'torch_geometric'` | `pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html` |
| `CUDA out of memory` | Reduce `--batch_size` to 8 or `--graph_hidden` to 32 |
| `DistilBERT download hangs` | Pre-download: `python -c "from transformers import DistilBertModel; DistilBertModel.from_pretrained('distilbert-base-uncased')"` |
| `Neo4j connection refused` | Start with `docker-compose up neo4j -d` |
| `No positive labels` | Check your data has action keywords (boycott, fired, etc.) |
| `DDP hangs` | Check `NCCL_DEBUG=INFO`, ensure all GPUs visible |

---

## File Structure After Setup

```
prsi/
├── src/
│   ├── data/              # Scrapers, preprocessing
│   ├── features/          # Graph extractor, sentiment, toxicity
│   ├── models/            # GNN, multimodal trainer, predictor
│   ├── attacks/           # Adversarial generator
│   ├── privacy/           # DP trainer, FL simulator
│   └── dashboard/         # Dashboard components
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── models/
│   └── multimodal_doom/   # Trained models
├── data/
│   └── processed_reddit_multimodal.csv
├── api_v2.py              # FastAPI v2
├── train_multimodal.py    # Training script
├── demo.py                # Viva demo
├── migrate.py             # v1 -> v2 migration
├── docker-compose.yml     # Docker orchestration
├── hpc_multimodal_train.sh # HPC PBS script
├── Makefile               # Common commands
└── requirements.txt       # Dependencies
```

---

## Next Steps

1. **Improve labels**: Manually annotate 500-1000 samples for better ground truth
2. **Add GNN edges**: Populate Neo4j with real user interactions
3. **Tune hyperparameters**: Use WandB sweeps on your H100 cluster
4. **Add real moderation proxy**: Integrate Perspective API for attack simulator
5. **Scale data**: Process more Pushshift months (target: 500k+ samples)

---

**You're ready. Go train that model.** 🔥
