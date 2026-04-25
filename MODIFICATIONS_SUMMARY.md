# Modifications Summary - April 26th

## ✅ What I Modified

### 1. `scripts/consolidate_datasets.py`
**Added:**
- `load_cade()` function to load CADE_aequa hate speech dataset
- Updated `consolidate_all_datasets()` to:
  - Prioritize CADE dataset (Priority 1)
  - Add HPC paths for all datasets
  - Skip multilingual violence dataset (no text content, only IDs)
  - Better error messages

**Why:** Your CADE dataset is high-quality labeled hate speech data perfect for doom score training.

### 2. `scripts/train_text_baseline.py`
**Modified:**
- Increased default batch size: 32 → 64 (optimized for H100)
- Increased learning rate: 2e-5 → 3e-5 (better for DistilBERT)
- Added `--num-workers` argument (default 8 for HPC DataLoader parallelism)
- Updated `find_latest_parquet()` to handle multiple naming patterns
- Added `pin_memory=True` for faster GPU transfer

**Why:** H100 can handle larger batches, and these settings will speed up training significantly.

## 📁 Files Already Present (No Changes Needed)

- `HPC_EXECUTION_GUIDE.md` - Complete step-by-step instructions ✅
- `data/raw/` and `data/processed/` directories exist ✅
- All other architecture files intact ✅

## 🚀 What You Need to Do on HPC

```bash
# 1. Copy datasets
cd /path/to/doom-index
mkdir -p data/raw
cp ~/CADE_aequa/data/*.parquet data/raw/
cp ~/cyberbullying-instagram-tiktok/*.parquet data/raw/
cp "~/hate/Tweets on Cancelled Brands.csv" data/raw/

# 2. Run consolidation
python scripts/consolidate_datasets.py --data-dir data/raw

# 3. Split into train/val/test (script provided in HPC_EXECUTION_GUIDE.md)

# 4. Train model
python scripts/train_text_baseline.py \
    --data-dir data/processed \
    --output-dir models/text_baseline \
    --batch-size 128 \
    --epochs 5 \
    --gpu 0 \
    --num-workers 8
```

## 🎯 Expected Results

- **Consolidation:** 30k-100k samples from CADE + Cyberbullying + Cancelled Brands
- **Training Time:** 2-4 hours on single H100
- **Expected Accuracy:** 78-85% validation, AUC > 0.85
- **Output:** Trained DistilBERT model ready for viva demo

## ⚠️ Key Decisions Made

1. **Skipped Multilingual Violence Dataset** - Only has tweet IDs, no actual text. Useless without Twitter API access to hydrate deleted/suspended accounts.

2. **Prioritized CADE** - High-quality hate speech labels, perfect for doom scoring.

3. **No Pushshift 2024-2026** - You have enough labeled data now. Don't wait for more.

4. **Text-only baseline first** - Get this working before adding multimodal complexity.

## 📅 Timeline

- **Today (April 26):** Copy data, run consolidation
- **Tonight:** Train model overnight (2-4 hours)
- **Tomorrow (April 27):** Verify results, test inference
- **April 28-May 6:** Polish, add features if time permits
- **May 7:** Viva ready!

**You're not starting from scratch. You're finishing what's 85% built.**
