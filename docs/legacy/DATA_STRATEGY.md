# PRSI Doom Index - Data Strategy & Training Guide

## Executive Summary

This document outlines the complete data strategy for the Predictive Reddit Social Index (PRSI) project, addressing the challenge of limited labeled cancellation data through semi-supervised learning techniques.

## Problem Statement

**Challenge**: No perfect dataset exists for predicting social media "cancellation" events.

**Solution**: Combine multiple related datasets (hate speech, cyberbullying, toxicity, sexism) with semi-supervised learning on large-scale unlabeled Pushshift Reddit data.

---

## Dataset Inventory

### Labeled Datasets (Supervised Training)

| Dataset | Samples | Platform | Label Type | Quality |
|---------|---------|----------|------------|---------|
| **Cyberbullying (IG/TikTok)** | ~50K | Instagram, TikTok | Binary (bullying/not) | High |
| **Hate Speech 1829** | ~1.8K | Multi-platform | Binary (hate/not) | High |
| **TweetBLM** | ~10K | Twitter | Binary (hate/not) | Medium-High |
| **Cancelled Brands** | ~5K | Twitter | Engagement-based | Medium |
| **Jigsaw Toxic** | ~160K | Web comments | Multi-label toxic | High |
| **Sexism Social Media** | ~14K | Social media | Binary (sexist/not) | High |
| **Doom Index (Custom)** | ~5K | Twitter | Cancellation signals | Medium |

**Total Unified Labeled**: ~50-80K samples after deduplication and balancing

### Unlabeled Datasets (Semi-Supervised)

| Dataset | Samples | Source | Usage |
|---------|---------|--------|-------|
| **Pushshift Reddit** | Millions | Reddit comments | Contrastive pre-training, pseudo-labeling |
| **Twitter Scraped** | 100K+ | Custom scraping | Domain adaptation |

---

## Data Unification Pipeline

### Architecture

```
Multiple Raw Datasets
        ↓
[Schema Normalization]
        ↓
[Text Cleaning & Validation]
        ↓
[Deduplication (SHA-256 hash)]
        ↓
[Stratified Balancing]
        ↓
[Train/Val/Test Split (80/10/10)]
        ↓
Unified Parquet Files
```

### Usage

```bash
# Run full unification pipeline
python -m src.data.unify_datasets \
    --data-root /path/to/datasets \
    --output-dir data/unified \
    --target-samples 10000

# Output files:
#   - data/unified/train.parquet
#   - data/unified/validation.parquet
#   - data/unified/test.parquet
#   - data/unified/all_data.parquet
#   - data/unified/dataset_report.json
```

### Key Features

1. **Schema Normalization**: Converts diverse schemas to unified format
   - `text`: Normalized text content
   - `label`: Binary (0=safe, 1=doom/toxic)
   - `source`: Original dataset identifier
   - `platform`: Social media platform
   - `engagement_score`: Normalized engagement metric
   - `metadata`: JSON with original features

2. **Quality Filters**
   - Minimum text length: 10 characters
   - Maximum text length: 512 characters
   - Valid labels only (0 or 1)
   - Duplicate removal via SHA-256 hashing

3. **Stratified Balancing**
   - Equal representation from each source
   - Balanced positive/negative classes
   - Preserves domain diversity

---

## Semi-Supervised Learning Strategies

### Strategy 1: Self-Training with Pseudo-Labeling ⭐ Recommended

**How it works:**
1. Train initial model on labeled data
2. Predict on unlabeled data
3. Keep high-confidence predictions (>95%)
4. Add pseudo-labeled samples to training set
5. Repeat for 3 iterations

**Command:**
```bash
python -m src.training.semi_supervised_trainer \
    --strategy self_training \
    --labeled-data data/unified/train.parquet \
    --val-data data/unified/validation.parquet \
    --unlabeled-data data/pushshift/unlabeled_texts.txt \
    --output-dir models/self_training \
    --epochs 5 \
    --confidence-threshold 0.95 \
    --n-iterations 3
```

**Expected Results:**
- Iteration 1: F1 = 0.75-0.80
- Iteration 2: F1 = 0.78-0.83 (+3-5%)
- Iteration 3: F1 = 0.80-0.85 (+2-3%)

### Strategy 2: Contrastive Pre-Training

**How it works:**
1. Pre-train encoder with contrastive loss on unlabeled data
2. Learn representations that cluster similar texts
3. Fine-tune classifier on labeled data

**Command:**
```bash
python -m src.training.semi_supervised_trainer \
    --strategy contrastive \
    --labeled-data data/unified/train.parquet \
    --val-data data/unified/validation.parquet \
    --unlabeled-data data/pushshift/unlabeled_texts.txt \
    --output-dir models/contrastive \
    --epochs 5
```

**Expected Results:**
- Better generalization to unseen domains
- Improved performance on rare cancellation patterns
- +2-4% F1 over supervised baseline

### Strategy 3: Supervised Only (Baseline)

**Command:**
```bash
python -m src.training.semi_supervised_trainer \
    --strategy supervised \
    --labeled-data data/unified/train.parquet \
    --val-data data/unified/validation.parquet \
    --output-dir models/supervised \
    --epochs 5
```

**Expected Results:**
- F1 = 0.72-0.78
- Good baseline for comparison

---

## Complete Training Workflow

### Step 1: Prepare Data

```bash
# Run automated pipeline
./scripts/run_data_pipeline.sh

# Or manual execution:
# 1. Unify labeled datasets
python -m src.data.unify_datasets \
    --data-root /home/vivek.120542 \
    --output-dir data/unified

# 2. Process Pushshift for unlabeled data
python -m src.data.pushshift_ingestion \
    --input /home/vivek.120542/RC_2023-01.zst \
    --output data/pushshift/reddit_unlabeled.parquet \
    --n-workers 16
```

### Step 2: Train Models

```bash
# Baseline supervised model
python -m src.training.semi_supervised_trainer \
    --strategy supervised \
    --output-dir models/baseline

# Best performing: Self-training
python -m src.training.semi_supervised_trainer \
    --strategy self_training \
    --output-dir models/final
```

### Step 3: Evaluate

```bash
python -m src.evaluation.evaluate_model \
    --model-path models/final/best \
    --test-data data/unified/test.parquet \
    --report-dir reports/evaluation
```

### Step 4: Deploy

```bash
# Export to ONNX for fast inference
python -m src.inference.export_onnx \
    --model-path models/final/best \
    --output models/doom_classifier.onnx

# Start API server
python -m src.api.api_v2 \
    --model-path models/doom_classifier.onnx \
    --port 8000
```

---

## Expected Performance Metrics

| Metric | Supervised | Self-Training | Contrastive |
|--------|-----------|---------------|-------------|
| **Accuracy** | 0.75-0.80 | 0.80-0.85 | 0.78-0.83 |
| **Precision** | 0.70-0.76 | 0.75-0.82 | 0.73-0.80 |
| **Recall** | 0.68-0.75 | 0.74-0.81 | 0.72-0.79 |
| **F1 Score** | 0.69-0.75 | 0.75-0.82 | 0.73-0.80 |
| **AUC-ROC** | 0.78-0.83 | 0.83-0.88 | 0.81-0.86 |

---

## Data Augmentation Techniques

### During Training

1. **Word Dropout**: Randomly remove 10-30% of words
2. **Word Swap**: Swap adjacent words (20% probability)
3. **Synonym Replacement**: Replace words with synonyms (future enhancement)
4. **Back Translation**: Translate to Hindi and back (for multilingual)

### For Adversarial Robustness

```bash
# Generate adversarial examples
python -m src.attacks.adversarial_production \
    --input data/unified/test.parquet \
    --output data/adversarial/test_adversarial.parquet \
    --attack-type textfooler \
    --num-examples 1000
```

---

## Handling Class Imbalance

The unified dataset may have imbalanced classes. Mitigation strategies:

1. **Oversampling**: Duplicate minority class samples
2. **Undersampling**: Reduce majority class
3. **Class Weights**: Apply higher loss weight to minority class
4. **Focal Loss**: Focus on hard-to-classify examples

```python
# In training config
class_weights = {0: 1.0, 1: 2.5}  # Weight positive class 2.5x
```

---

## Quality Assurance

### Data Validation Checks

```bash
python -m src.validation.data_validator \
    --dataset data/unified/train.parquet \
    --report-dir reports/validation
```

**Checks performed:**
- ✅ Label distribution balance
- ✅ Text length distribution
- ✅ Duplicate detection
- ✅ PII leakage detection
- ✅ Engagement score validity
- ✅ Source diversity

### Model Validation

- Cross-validation across sources
- Per-source performance breakdown
- Confusion matrix analysis
- Error case inspection

---

## Troubleshooting

### Issue: Low pseudo-label generation rate

**Symptoms**: <10% of unlabeled data receives pseudo-labels

**Solutions:**
1. Lower confidence threshold (0.95 → 0.85)
2. Train supervised model longer before pseudo-labeling
3. Use ensemble of models for prediction

### Issue: Overfitting on small labeled set

**Symptoms**: High train accuracy, low validation accuracy

**Solutions:**
1. Increase regularization (weight decay)
2. Reduce model size (DistilBERT → TinyBERT)
3. More aggressive data augmentation
4. Earlier stopping

### Issue: Domain shift between sources

**Symptoms**: Poor performance on specific sources

**Solutions:**
1. Domain-adversarial training
2. Per-source fine-tuning
3. Multi-task learning with source as auxiliary task

---

## Future Enhancements

1. **Multilingual Support**: Add Hindi/Hinglish datasets
2. **Multimodal**: Incorporate meme/image datasets (MemeLens)
3. **Temporal Modeling**: Time-series features for doom trajectory
4. **Active Learning**: Human-in-the-loop for ambiguous cases
5. **Continual Learning**: Update model with new cancellation events

---

## References

- DistilBERT: https://arxiv.org/abs/1910.01108
- Self-Training Survey: https://arxiv.org/abs/2109.14033
- SimCLR: https://arxiv.org/abs/2002.05709
- Jigsaw Toxic Comments: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

---

## Contact & Support

For issues or questions:
- GitHub Issues: https://github.com/killer1panda/prsi/issues
- Documentation: `/workspace/docs/`
