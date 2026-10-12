# Quick Start: Data Pipeline for PRSI Doom Index

## TL;DR - Run Everything in One Command

```bash
./scripts/run_data_pipeline.sh
```

This will:
1. ✅ Unify all your labeled datasets into train/val/test splits
2. ✅ Process Pushshift Reddit data for semi-supervised learning
3. ✅ Generate comprehensive statistics report

---

## Your Datasets (Already Available)

Based on your inventory, we'll use:

### Supervised (Labeled):
- ✅ `cyberbullying-instagram-tiktok/train.parquet` - Instagram/TikTok bullying
- ✅ `hate/hate_speech_1829.csv` - Multi-platform hate speech
- ✅ `hate/TweetBLM.csv` - Twitter BLM-related hate
- ✅ `hate/Tweets on Cancelled Brands.csv` - Brand cancellation tweets
- ✅ `hate/archive (7)/train.csv` - Jigsaw toxic comments
- ✅ `hate/Social-Media-Toxic-Comments-Classification-main/data/train.csv` - More toxic comments
- ✅ `~/tum-nlp-sexism-socialmedia-balanced/sexism-socialmedia-balanced.csv` - Sexism detection
- ✅ `doom-index/processed_sample.csv` - Your custom doom index data

### Unsupervised (Unlabeled):
- ✅ `RC_2023-01.zst` - Pushshift Reddit comments (millions of samples)

---

## Step-by-Step Execution

### Option 1: Automated Pipeline (Recommended)

```bash
# Navigate to project root
cd /workspace

# Run the full pipeline
./scripts/run_data_pipeline.sh
```

**Expected Output:**
```
==========================================
PRSI Doom Index - Data Pipeline
==========================================
Data Root: /home/vivek.120542
Output Directory: data/unified
Target Samples per Source: 10000

[Step 1/3] Unifying labeled datasets...
INFO: Loading Cyberbullying...
INFO: Loaded cyberbullying dataset: 47832 samples
INFO: Loading Hate Speech...
INFO: Loaded hate_speech_1829: 1829 samples
INFO: Loaded TweetBLM: 9847 samples
...
INFO: Deduplication: removed 3421 duplicates, kept 78234 samples
INFO: Balancing datasets...
INFO: Splits created: train=62587, val=7823, test=7824

[Step 2/3] Preparing unlabeled Pushshift data...
Found Pushshift archive, processing...
INFO: Processing 2.3M Reddit comments...

[Step 3/3] Generating dataset statistics...

Dataset Statistics:
============================================================
TRAIN:
  Total: 62,587
  Positive: 31,294 (50.0%)
  Negative: 31,293
  Sources: cyberbullying_instagram_tiktok, hate_speech_1829, ...
  Avg Text Length: 127.3

VALIDATION:
  Total: 7,823
  ...

TEST:
  Total: 7,824
  ...

==========================================
✅ Data pipeline completed successfully!
==========================================
```

### Option 2: Manual Execution

#### Step 1: Unify Labeled Datasets

```bash
python -m src.data.unify_datasets \
    --data-root /home/vivek.120542 \
    --output-dir data/unified \
    --target-samples 10000 \
    --min-text-length 10 \
    --max-text-length 512
```

**Output Files:**
- `data/unified/train.parquet` (60-80K samples)
- `data/unified/validation.parquet` (7-10K samples)
- `data/unified/test.parquet` (7-10K samples)
- `data/unified/all_data.parquet` (combined)
- `data/unified/dataset_report.json` (statistics)

#### Step 2: Prepare Unlabeled Data (Optional but Recommended)

```bash
python -m src.data.pushshift_ingestion \
    --input /home/vivek.120542/RC_2023-01.zst \
    --output data/pushshift/reddit_unlabeled.parquet \
    --n-workers 16 \
    --mode unlabeled_only
```

This extracts plain text from Reddit comments for semi-supervised learning.

#### Step 3: Train Semi-Supervised Model

```bash
# Self-training (best performance)
python -m src.training.semi_supervised_trainer \
    --strategy self_training \
    --labeled-data data/unified/train.parquet \
    --val-data data/unified/validation.parquet \
    --unlabeled-data data/pushshift/reddit_unlabeled.parquet \
    --output-dir models/semi_supervised \
    --epochs 5 \
    --confidence-threshold 0.95 \
    --n-iterations 3
```

**Training Time:** ~2-4 hours on single GPU (RTX 3090/4090)

---

## What You'll Get

### Unified Dataset Structure

Each parquet file contains:

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Normalized text content |
| `label` | int64 | Binary label (0=safe, 1=doom/toxic) |
| `source` | string | Original dataset name |
| `sample_id` | string | Unique SHA-256 hash |
| `language` | string | Language code (default: "en") |
| `platform` | string | Social media platform |
| `engagement_score` | float64 | Normalized engagement metric |
| `metadata` | string | JSON with original features |

### Model Checkpoints

After training:
```
models/semi_supervised/
├── iter_0/           # First iteration
│   ├── model/
│   └── tokenizer/
├── iter_1/           # Second iteration (with pseudo-labels)
│   ├── model/
│   └── tokenizer/
├── iter_2/           # Third iteration
│   ├── model/
│   └── tokenizer/
└── best/             # Best model by F1 score
    ├── model/
    ├── tokenizer/
    └── training_args.json
```

---

## Verification

### Check Dataset Quality

```bash
python -c "
import pandas as pd
from pathlib import Path

df = pd.read_parquet('data/unified/train.parquet')
print(f'Total samples: {len(df):,}')
print(f'Positive class: {(df.label == 1).sum():,} ({(df.label == 1).mean()*100:.1f}%)')
print(f'Sources: {df.source.nunique()}')
print(f'Platforms: {df.platform.unique().tolist()}')
print(f'Avg text length: {df.text.str.len().mean():.1f}')
"
```

### Test Model Inference

```bash
python -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained('models/semi_supervised/best')
tokenizer = AutoTokenizer.from_pretrained('models/semi_supervised/best')

text = 'This is a test tweet about controversy'
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    
print(f'Doom Score: {probs[0, 1].item()*100:.2f}%')
print(f'Prediction: {'DOOM' if probs[0, 1] > 0.5 else 'SAFE'}')
"
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:** Reduce batch size
```bash
python -m src.training.semi_supervised_trainer \
    --batch-size 16 \  # or even 8
    ...
```

### Issue: No Pseudo-Labels Generated

**Solution:** Lower confidence threshold
```bash
--confidence-threshold 0.85  # instead of 0.95
```

### Issue: Missing Dataset Files

**Solution:** Update paths to match your actual locations
```bash
# Check what exists
ls -la /home/vivek.120542/*.parquet
ls -la /home/vivek.120542/hate/*.csv
```

---

## Next Steps After Training

1. **Evaluate on Test Set**
   ```bash
   python -m src.evaluation.evaluate_model \
       --model-path models/semi_supervised/best \
       --test-data data/unified/test.parquet \
       --report-dir reports/final_evaluation
   ```

2. **Export for Production**
   ```bash
   python -m src.inference.export_onnx \
       --model-path models/semi_supervised/best \
       --output models/doom_classifier.onnx
   ```

3. **Start API Server**
   ```bash
   python -m src.api.api_v2 \
       --model-path models/doom_classifier.onnx \
       --port 8000
   ```

4. **Run Viva Demo**
   ```bash
   python scripts/run_pipeline_v2.py --demo --forecast
   ```

---

## Performance Expectations

| Strategy | Training Time | Expected F1 | GPU Memory |
|----------|--------------|-------------|------------|
| Supervised Only | 30 min | 0.72-0.78 | 8GB |
| Self-Training (3 iter) | 2-3 hrs | 0.78-0.85 | 8GB |
| Contrastive + Fine-tune | 4-5 hrs | 0.76-0.83 | 12GB |

---

## Additional Resources

- Full documentation: `DATA_STRATEGY.md`
- Model architecture: `src/models/fusion.py`
- Adversarial testing: `src/attacks/adversarial_production.py`
- Deployment guide: `docs/deployment.md`

---

## Support

If you encounter issues:
1. Check logs in `logs/` directory
2. Review `reports/validation/` for data quality issues
3. Open GitHub issue: https://github.com/killer1panda/prsi/issues
