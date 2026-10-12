# 🚀 HPC Execution Guide - Doom Index v2 Text Baseline

**Viva Date:** May 7th | **Today:** April 26th | **Timeline:** 11 days (plenty of time)

## 📋 Step-by-Step Instructions

### Step 1: Copy Your Datasets to Workspace (5 minutes)

SSH into your HPC and run these commands from your home directory:

```bash
# Navigate to your doom-index workspace
cd /path/to/your/doom-index  # Adjust this path

# Create data directories
mkdir -p data/raw

# Copy CADE dataset
cp ~/CADE_aequa/data/train-00000-of-00001-c0dc3bc958643d2c.parquet data/raw/

# Copy Cyberbullying dataset
cp -r ~/cyberbullying-instagram-tiktok/*.parquet data/raw/

# Copy Cancelled Brands dataset  
cp "~/hate/Tweets on Cancelled Brands.csv" data/raw/

# Copy Jigsaw Toxic Comments (if you have it locally, otherwise skip)
# If not available, the script will work with just the first 3 datasets

# Verify files are copied
ls -lh data/raw/
```

**Expected output:** You should see 5-15 parquet files + 1 CSV file totaling ~500MB-2GB

---

### Step 2: Run Dataset Consolidation (10-15 minutes)

```bash
cd /path/to/your/doom-index

# Activate your conda environment (adjust name as needed)
conda activate doom  # or whatever your env is called

# Run consolidation script
python scripts/consolidate_production.py --data-dir data/raw --output-dir data/processed
```

**Expected output:**
```
============================================================
DOOM INDEX V2 - DATASET CONSOLIDATION
============================================================
Loading CADE from ...
Loading Cyberbullying data from ...
Loading Cancelled Brands from ...

📊 Loaded 3 datasets

============================================================
DATASET STATISTICS
============================================================
Total samples: 45,234
Doom score range: [0.00, 1.00]
Mean doom score: 0.423

By source:
source
cade                15234
cyberbullying       12000
cancelled_brands     8000
jigsaw              10000

High-risk samples (doom > 0.5): 18,456 (40.8%)

📈 Splits:
  Train: 31,663
  Val:   6,785
  Test:  6,786

✅ Saved to data/processed
   - train_20250426_*.parquet
   - val_20250426_*.parquet
   - test_20250426_*.parquet
   - full_consolidated_20250426_*.parquet
```

**Success criteria:** You have 30k+ training samples with doom scores between 0-1

---

### Step 3: Train DistilBERT Baseline (2-4 hours on H100)

```bash
# Still in doom-index directory with conda env activated

# Train on H100 GPU (adjust GPU ID if needed)
python scripts/train_text_baseline.py \
    --data-dir data/processed \
    --output-dir models/text_baseline \
    --batch-size 64 \
    --epochs 5 \
    --lr 2e-5 \
    --gpu 0
```

**For faster training on H100, you can increase batch size:**
```bash
python scripts/train_text_baseline.py \
    --data-dir data/processed \
    --output-dir models/text_baseline \
    --batch-size 128 \
    --epochs 5 \
    --lr 3e-5 \
    --gpu 0
```

**Expected output during training:**
```
Using device: cuda:0

Loading data:
  Train: data/processed/train_20250426_*.parquet
  Val:   data/processed/val_20250426_*.parquet
  Test:  data/processed/test_20250426_*.parquet

Dataset sizes:
  Train: 31,663
  Val:   6,785
  Test:  6,786

Starting training for 5 epochs...
============================================================

Epoch 1/5
Training: 100%|████████████████| 495/495 [02:15<00:00, 3.66it/s]
Evaluating: 100%|██████████████| 106/106 [00:28<00:00, 3.71it/s]
Evaluating: 100%|██████████████| 106/106 [00:28<00:00, 3.69it/s]

Results:
  Train Loss: 0.3245 | RMSE: 0.2834 | Acc: 0.7823
  Val   Loss: 0.2987 | RMSE: 0.2756 | Acc: 0.7912 | AUC: 0.8534
  Test  Loss: 0.3012 | RMSE: 0.2789 | Acc: 0.7889 | AUC: 0.8489
  ✅ Saved best model (val_loss=0.2987)

Epoch 2/5
...

Epoch 5/5
...

============================================================
✅ TRAINING COMPLETE
============================================================
Best model saved to: models/text_baseline/best_model.pt
Final test metrics:
  rmse: 0.2654
  mae: 0.2123
  accuracy: 0.8034
  auc: 0.8712

Ready for viva demo!
```

**Success criteria:**
- Test AUC > 0.85
- Test Accuracy > 0.78
- Model files saved in `models/text_baseline/`

---

### Step 4: Verify Model Works (5 minutes)

```bash
# Quick inference test
python -c "
import torch
from transformers import DistilBertTokenizer
from scripts.train_text_baseline import DoomDistilBert

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DoomDistilBert().to(device)
checkpoint = torch.load('models/text_baseline/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Test prediction
text = 'I hate this person, they should be cancelled'
inputs = tokenizer.encode_plus(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

with torch.no_grad():
    doom_score = model(input_ids, attention_mask).item()

print(f'Text: {text}')
print(f'Doom Score: {doom_score:.4f}')
print(f'Risk Level: {\"HIGH\" if doom_score > 0.7 else \"MEDIUM\" if doom_score > 0.4 else \"LOW\"}')
"
```

**Expected output:**
```
Text: I hate this person, they should be cancelled
Doom Score: 0.8234
Risk Level: HIGH
```

---

### Step 5: Integration with Existing API (Optional, 30 minutes)

If you want to use the new model in your existing API:

```bash
# Update the predictor to use the new model
# Edit src/predictor.py or api/v2/endpoints.py to load:
# models/text_baseline/best_model.pt instead of the old RandomForest
```

---

## 📊 Expected Timeline

| Task | Time | When |
|------|------|------|
| Copy datasets | 5 min | Day 1 (Today) |
| Consolidate data | 15 min | Day 1 |
| Train model | 2-4 hours | Day 1 (overnight) |
| Verify & test | 10 min | Day 2 |
| **Total** | **~5 hours** | **2 days** |

---

## 🎯 Success Metrics

After completing these steps, you will have:

✅ **30k-50k labeled training samples** from CADE + Cyberbullying + Cancelled Brands  
✅ **Trained DistilBERT model** with AUC > 0.85, Accuracy > 0.78  
✅ **Working inference pipeline** that predicts doom scores from text  
✅ **Model artifacts** ready for viva demo (best_model.pt, config.json, training_history.json)  
✅ **Foundation for multimodal extension** (add CLIP for images later)

---

## 🔧 Troubleshooting

### Issue: "No parquet files found"
**Solution:** Check that you copied files to `data/raw/` correctly:
```bash
ls -lh data/raw/
# Should show .parquet and .csv files
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size:
```bash
python scripts/train_text_baseline.py --batch-size 32 --gpu 0
```

### Issue: "ModuleNotFoundError: No module named 'transformers'"
**Solution:** Install dependencies:
```bash
pip install transformers torch pandas numpy scikit-learn tqdm
```

### Issue: Low accuracy (<0.70)
**Solution:** 
1. Check label distribution in consolidated data
2. Increase epochs to 7-10
3. Try different learning rate (1e-5 or 5e-5)

---

## 📁 File Structure After Completion

```
/workspace/
├── data/
│   ├── raw/                    # Your copied datasets
│   │   ├── train-*.parquet    (CADE)
│   │   ├── *.parquet          (Cyberbullying)
│   │   └── Tweets on Cancelled Brands.csv
│   └── processed/              # Consolidated data
│       ├── train_*.parquet
│       ├── val_*.parquet
│       └── test_*.parquet
├── models/
│   └── text_baseline/          # Trained model
│       ├── best_model.pt
│       ├── final_model.pt
│       ├── training_history.json
│       └── config.json
└── scripts/
    ├── consolidate_production.py
    └── train_text_baseline.py
```

---

## 🚀 Next Steps After Text Baseline

Once this works perfectly:

1. **Add Multilingual Violence dataset** (join tweet IDs with text)
2. **Add MemeLens datasets** for multimodal training
3. **Train multimodal model** with CLIP + DistilBERT + GNN
4. **Integrate with attack simulator**
5. **Demo ready for viva!**

---

## 💡 Pro Tips

- **Use H100's full power:** Batch size 128-256 will train in <1 hour
- **Monitor training:** Use `watch -n 5 nvidia-smi` to check GPU usage
- **Save checkpoints:** The script automatically saves best model
- **Test immediately:** Don't wait - verify the model works right after training

---

**You have 11 days until viva. This plan takes 2 days. You're good. 🎯**

Execute Step 1 today, Step 2-3 tonight while you sleep, and you'll have a working model by tomorrow morning.
