# DOOM INDEX v2 - DATA INTEGRATION PLAN
## Based on Dataset Inventory Analysis (7,625 files)

---

## 🎯 EXECUTIVE SUMMARY

You have **excellent labeled data** (30+ high-quality datasets, ~50k-100k samples potential). 
**STOP planning complex pretraining schemes.** Your path to viva readiness:

1. **Consolidate existing labeled datasets** (Week 1)
2. **Train baseline model** (Week 2)  
3. **Add multimodal components** (Week 3-4)

---

## 📊 YOUR ACTUAL ASSETS

### Tier 1: Immediate Use (Highest Priority)
| Dataset | Format | Est. Samples | Label Type | Action |
|---------|--------|--------------|------------|--------|
| Tweets on Cancelled Brands | CSV | ~5-10k | Binary (cancelled/not) | **USE FIRST** |
| Multilingual Twitter Collective Violence | 12x Parquet | ~20-30k | Temporal sentiment | Map to doom scores |
| Cyberbullying Instagram/TikTok | Parquet | ~10k | Binary/Multi-class | Direct use |
| Jigsaw Toxic Comments | CSV | ~100k+ | 6 toxicity dimensions | Aggregate to single score |
| cancellation_events.csv | CSV | ~1-5k | Curated events | Ground truth |
| **MemeLens (Multi3Hate, MAMI, FHM)** | **Parquet** | **~20-30k** | **Hateful meme binary** | **Multimodal training** |

### Tier 2: Augmentation (Secondary)
- WELFake (fake news detection)
- TwitterAAE (annotated tweets with .ann files)
- Sexism social media balanced
- Hate speech corpora (multiple)
- ProvocationProbe

### Tier 3: Discard/Low Priority
- Reddit 2008-2012 dumps (outdated cultural context)
- Most Pushshift RC_2023 files (incomplete, no labels)
- Any JSON files with parse errors

---

## 🚫 WHAT NOT TO DO

❌ **Don't use GPT-2/XL/LLMs for labeling** - You already agreed to skip this ✓
❌ **Don't chase more Pushshift data** - Your labeled datasets are sufficient
❌ **Don't train unsupervised first** - Waste of time with this much labeled data
❌ **Don't over-engineer data pipeline** - Start simple, iterate

---

## ✅ CONCRETE ACTION PLAN

### Phase 1: Data Consolidation (Days 1-3)

**Goal:** Create unified training dataset from existing labeled sources

```bash
# Step 1: Copy all source files to workspace
mkdir -p /workspace/data/raw
# Copy from your paths:
# - /home/vivek.120542/hate/*.csv
# - /home/vivek.120542/multilingual-twitter-collective-violence-dataset/data/*.parquet
# - /home/vivek.120542/cyberbullying-instagram-tiktok/*.parquet
# - /home/vivek.120542/doom-index/doom index/data/twitter_dataset/*.csv
```

**Create unified schema:**
```python
unified_schema = {
    'text': str,              # Tweet/comment text
    'label': float,           # 0.0-1.0 doom score (or binary)
    'source': str,            # Dataset origin
    'timestamp': datetime,    # When posted
    'engagement': dict,       # likes, retweets, replies
    'user_features': dict,    # followers, verified status
    'media': list,            # Images/videos if any
}
```

**Script to write:** `scripts/consolidate_datasets.py`
- Load each Tier 1 dataset
- Normalize labels to 0-1 scale
- Handle class imbalance
- Export as `data/unified_train.parquet`

### Phase 2: Label Normalization Strategy (Day 4)

**Problem:** Different datasets have different label schemes
**Solution:** Map everything to "Doom Score" (0-1)

| Original Label | Mapping | Rationale |
|----------------|---------|-----------|
| Cancelled Brands (binary) | 1.0 = cancelled, 0.0 = not | Direct mapping |
| Jigsaw (6 columns) | max(toxic, severe_toxic, identity_hate) | Worst-case toxicity |
| Cyberbullying (binary) | Direct 0/1 | Already aligned |
| Collective Violence | post7geo70 / 100 | Peak outrage metric |
| Hate speech | 1.0 = hate, 0.5 = offensive, 0.0 = clean | Graduated scale |

**Weak supervision rules** (no ML needed):
```python
def compute_doom_score(row):
    score = 0.0
    
    # Rule 1: High negative engagement spike
    if row['negative_replies_24h'] > 1000:
        score += 0.3
    
    # Rule 2: Media coverage detected
    if row['has_news_coverage']:
        score += 0.25
    
    # Rule 3: Boycott/outrage hashtags
    if any(tag in row['hashtags'] for tag in ['#boycott', '#outrage', '#cancel']):
        score += 0.2
    
    # Rule 4: Rapid follower loss
    if row['follower_change_24h'] < -0.1:  # 10% loss
        score += 0.25
    
    return min(score, 1.0)
```

### Phase 3: Baseline Training (Days 5-7)

**Forget multimodal initially.** Train a simple but strong baseline:

```python
# models/doom_baseline.py
from sklearn.ensemble import RandomForest, GradientBoosting
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load unified data
df = pd.read_parquet('data/unified_train.parquet')

# Text features
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_text = tfidf.fit_transform(df['text'])

# Metadata features
X_meta = df[['followers', 'verified', 'retweets', 'replies']].values

# Combine
from scipy.sparse import hstack
X = hstack([X_text, X_meta])
y = df['doom_score']

# Train
model = GradientBoosting(n_estimators=200, max_depth=6)
model.fit(X, y)

# Evaluate
from sklearn.metrics import roc_auc_score, mean_squared_error
y_pred = model.predict(X)
print(f"AUC: {roc_auc_score(y > 0.5, y_pred)}")
```

**Target metrics:**
- AUC-ROC: >0.85
- Accuracy (binary): >80%
- Inference time: <100ms per sample

### Phase 4: Multimodal Upgrade (Week 2-3)

**Only after baseline works**, add:

1. **DistilBERT for text** (replace TF-IDF)
   - Fine-tune on your unified dataset
   - Extract [CLS] embeddings (768 dim)

2. **GraphSAGE for user networks** (if Neo4j populated)
   - Build reply/mention graph
   - Aggregate neighbor features

3. **CLIP/ViT for images** (if you collect meme dataset)
   - Detect viral meme templates
   - Extract visual embeddings

**Fusion architecture:**
```
Text (768) ──┐
             ├──> Concat -> Dense(512) -> Dense(256) -> Doom Score
Graph (128) ─┤
Image (512) ─┘
```

### Phase 5: Full Pipeline Integration (Week 4)

Update existing scripts:
- `train_model_full_fixed.py` → use unified dataset
- `build_neo4j_graph.py` → populate with real data
- `api_v2.py` → load trained model
- `dashboard/app.py` → show real predictions

---

## 📁 FILE STRUCTURE TO CREATE

```
/workspace/
├── data/
│   ├── raw/                    # Copied from your home directory
│   │   ├── cancelled_brands.csv
│   │   ├── collective_violence/*.parquet
│   │   ├── cyberbullying.parquet
│   │   └── jigsaw_toxic.csv
│   ├── processed/
│   │   └── unified_train.parquet   # Your golden dataset
│   └── external/               # Future additions
├── scripts/
│   ├── consolidate_datasets.py     # NEW - Phase 1
│   ├── normalize_labels.py         # NEW - Phase 2
│   └── validate_data.py            # NEW - Quality checks
├── models/
│   ├── doom_baseline.pkl           # Phase 3 output
│   ├── distilbert_finetuned/       # Phase 4 output
│   └── multimodal_doom/            # Final model
└── notebooks/
    └── data_exploration.ipynb      # Understand your data
```

---

## ⚠️ CRITICAL WARNINGS

1. **Data location:** All your datasets are in `/home/vivek.120542/` - need to copy to workspace
2. **Class imbalance:** Cancellation is rare (~5-10% positive). Use:
   - Class weights in training
   - Focal loss
   - Oversampling minority class
3. **Temporal leakage:** Ensure train/test split by time, not randomly
4. **Multilingual:** Your collective violence dataset has multiple languages
   - Use multilingual BERT or filter to English only initially

---

## 🎯 SUCCESS METRICS

**Phase 1 Complete when:**
- [ ] All Tier 1 datasets copied to `/workspace/data/raw/`
- [ ] `unified_train.parquet` created with >20k samples
- [ ] Label distribution: 15-25% positive (doom > 0.5)

**Phase 2 Complete when:**
- [ ] Label normalization script working
- [ ] Weak supervision rules validated on 100 manual samples

**Phase 3 Complete when:**
- [ ] Baseline model achieves AUC > 0.85 on held-out test set
- [ ] Inference time < 100ms
- [ ] Model saved and loadable in API

**Viva Ready when:**
- [ ] End-to-end demo: username → Doom Score in <5 seconds
- [ ] Attack simulator generates adversarial examples
- [ ] Dashboard shows live predictions
- [ ] Neo4j graph populated with sample network

---

## 🔧 IMMEDIATE NEXT STEPS (Do Today)

1. **Copy datasets from your home directory:**
```bash
cp /home/vivek.120542/hate/"Tweets on Cancelled Brands.csv" /workspace/data/raw/
cp /home/vivek.120542/hate/cancellation_events.csv /workspace/data/raw/
# ... copy all Tier 1 datasets
```

2. **Create consolidation script** (I'll help write this)

3. **Run initial EDA** to understand label distributions

4. **Train baseline within 48 hours** - don't overthink it

---

## 💡 FINAL ADVICE

You have **better data than 90% of ML projects**. The bottleneck is not data quantity or quality—**it's execution**. 

Stop planning. Start copying files and training models. A simple model on good data beats a complex model on hypothetical data every time.

Your infrastructure is world-class. Now feed it real data.

---

## 🎬 UPDATED RECOMMENDATION BASED ON MEME DATASETS

**Revised Strategy:**

1. **Week 1:** Consolidate text datasets + train baseline (RandomForest/GradientBoosting)
2. **Week 2:** Add MemeLens images → train CLIP + text fusion model
3. **Week 3:** Fine-tune DistilBERT, add GraphSAGE if Neo4j populated
4. **Week 4:** Full integration + viva rehearsal

**You can skip the "collect image data" phase entirely** because MemeLens has ~30k labeled hateful memes with embedded images. This accelerates your timeline by 2-3 weeks.
