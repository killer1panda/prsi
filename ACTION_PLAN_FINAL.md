# HONEST DATA STRATEGY & ACTION PLAN
## Based on Your Actual Dataset Inventory (1,131 files analyzed)

---

## 🎯 WHAT YOU ACTUALLY HAVE (The Real Assets)

### Tier 1: GOLD STANDARD (Use These First)
| Dataset | Format | Est. Samples | Why It's Valuable |
|---------|--------|--------------|-------------------|
| **Tweets on Cancelled Brands.csv** | CSV | ~5-10k | **ONLY true cancellation ground truth** - brand names, dates, actual cancellation events |
| **cancellation_events.csv** | CSV | Unknown | Curated cancellation cases (need to inspect) |
| **cyberbullying-instagram-tiktok/train.parquet** | Parquet | ~10k | Text + labels, platform-specific |
| **Jigsaw Toxic Comments** (archive 7, Social-Media-Toxic) | CSV | ~100k+ | Multi-label toxicity, industry standard |
| **MemeLens MAMI** (stereotype_en, objectification_en) | Parquet w/ images | ~10k | **Multimodal ready** - memes with text + image bytes |
| **MemeLens Multi3Hate** (Hateful_zh) | Parquet w/ images | ~5k | Multimodal hateful memes |
| **MemeLens FHM** (Hateful_ar, Hateful_en) | Parquet w/ images | ~3k | Multimodal training |
| **MIMIC2024** (Misogyny_hi_en) | Parquet w/ images | ~2k | **2024 data**, Hindi-English code-mixed memes |

### Tier 2: GOOD BUT NEEDS WORK
| Dataset | Issue | Fix Required |
|---------|-------|--------------|
| **Multilingual Twitter Collective Violence** | Only tweet IDs + geo features, NO text | You said you have both files on HPC - verify the tweet text file exists |
| **TweetBLM.csv** | Has text + hate label | Good for BLM-specific toxicity, merge with main corpus |
| **hate_speech_1829.csv** | Has text + label | Standard hate speech, useful for pretraining |
| **ProvocationProbe.csv** | tweet_id + label + topic | Need to check if text is included |
| **WELFake_Dataset.csv** | title + text + label | Fake news detection, adjacent use case |
| **sexism-socialmedia-balanced.csv** | Text + sexist label | Good for gender-based doom scoring |

### Tier 3: PUSHSHIFT DATA (Your Question)
```
FILE: /home/vivek.120542/RC_2023-01.zst  ← ONLY ONE 2023 FILE
FILES: RC_2008-12.* (multiple years)     ← CULTURALLY OBSOLETE
```

**HONEST ASSESSMENT OF PUSHSHIFT:**
- ❌ **Only 1 file from 2023** (January only)
- ❌ **Rest are 2008-2012** - "cancellation culture" didn't exist then
- ❌ **No labels** - raw comments without ground truth
- ❌ **Wrong cultural context** - 2010 Reddit ≠ 2026 Twitter
- ✅ **You claim access to 2024-2026 files** - IF TRUE, this changes everything

**VERIFICATION NEEDED:**
```bash
# Run this on your HPC NOW:
ls -lh /home/vivek.120542/RC_202*.zst
ls -lh /home/vivek.120542/RC_2024*.zst
ls -lh /home/vivek.120542/RC_2025*.zst
ls -lh /home/vivek.120542/RC_2026*.zst
```

If you truly have 2024-2026 Pushshift files, **this is gold**. If not, stop planning around them.

### Tier 4: RAW CORPUS (basegrande*.txt files)
- 20 files, ~20MB each = **~400MB of raw text**
- No labels, unknown quality
- **Use for**: Language model pretraining ONLY after supervised fine-tuning
- **Don't use for**: Direct doom score training (no labels)

---

## 🚫 WHAT YOU DON'T HAVE (Critical Gaps)

### ❌ CADE Dataset
- **Not found in inventory** despite your claim
- If it exists on HPC, provide exact path
- If not, stop mentioning it

### ❌ Recent 2024-2026 Cancellation Events
- Only MIMIC2024 memes found
- No verified 2024-2026 Twitter/X cancellation threads
- **Fix**: Scrape r/SubredditDrama, r/OutOfTheLoop from 2024-2026

### ❌ Sufficient Image-Text Pairs Outside MemeLens
- MemeLens gives you ~20k multimodal samples
- For robust multimodal training, you need **50k+**
- **Sources to add**:
  - Hateful Memes Challenge datasets (Facebook)
  - Hatemoji (hateful emoji + image datasets)
  - Custom scrape: Twitter image posts from cancellation threads

---

## 🔥 HONEST RECOMMENDATIONS

### Phase 1: STOP PLANNING, START TRAINING (Week 1)
**Goal**: Get a working baseline on existing labeled data

```python
# Datasets to consolidate IMMEDIATELY:
datasets_phase1 = [
    "/home/vivek.120542/hate/Tweets on Cancelled Brands.csv",  # Ground truth
    "/home/vivek.120542/cyberbullying-instagram-tiktok/train.parquet",
    "/home/vivek.120542/hate/archive (7)/train.csv",  # Jigsaw
    "/home/vivek.120542/hate/Social-Media-Toxic-Comments-Classification-main/data/train.csv",
    "/home/vivek.120542/hate/hate_speech_1829.csv",
    "/home/vivek.120542/hate/TweetBLM.csv",
    "/home/vivek.120542/~/tum-nlp-sexism-socialmedia-balanced/sexism-socialmedia-balanced.csv",
]
# Expected: 30-50k labeled text samples
```

**Action**: 
1. Copy these 7 datasets to `/workspace/data/raw/`
2. Run `scripts/consolidate_datasets.py` (I'll create this)
3. Train RandomForest baseline TODAY
4. Target: 80%+ accuracy on held-out test set

### Phase 2: Multimodal Training (Week 2-3)
**Goal**: Add image understanding with MemeLens

```python
datasets_phase2 = [
    "/home/vivek.120542/MemeLens/stereotype_en__MAMI/*.parquet",      # ~10k memes
    "/home/vivek.120542/MemeLens/Hateful_zh__Multi3Hate/*.parquet",   # ~5k memes  
    "/home/vivek.120542/MemeLens/Hateful_ar__Prop2Hate-Meme/*.parquet", # ~3k memes
    "/home/vivek.120542/MemeLens/Misogyny_hi_en__MIMIC2024/*.parquet", # 2024 data!
]
# Expected: 20k image-text pairs with labels
```

**Action**:
1. Extract image bytes from parquet files
2. Save images to `/workspace/data/images/`
3. Create multimodal dataset loader (already exists in `src/data/multimodal_dataset.py`)
4. Fine-tune CLIP + DistilBERT fusion
5. Target: AUC > 0.85 on meme toxicity

### Phase 3: Pushshift Decision Point (Week 3)
**CRITICAL**: Verify your 2024-2026 claim

```bash
# Run on HPC and share output:
find /home/vivek.120542 -name "RC_202*.zst" -o -name "RC_2024*.zst" -o -name "RC_2025*.zst" -o -name "RC_2026*.zst" | head -50
```

**IF you have 2024-2026 files (10+ files):**
- ✅ Worth processing for unsupervised pretraining
- Use for domain-adaptive pretraining (DAPT) of DistilBERT
- NOT for direct labeling - use weak supervision instead

**IF you only have 2008-2023 files:**
- ❌ **ABANDON PUSHSHIFT ENTIRELY**
- Cultural mismatch will hurt more than help
- Focus on augmenting existing labeled data instead

### Phase 4: Graph Features (Week 4)
**Goal**: Build Neo4j graph from interaction data

```python
# Sources for user interaction graphs:
graph_sources = [
    "enhanced_tweets.csv",  # Has replies, retweets
    "replies.csv",
    "timeline_tweets.csv",
]
```

**Action**:
1. Run `build_neo4j_graph_production.py`
2. Extract user-user interaction networks
3. Compute echo-chamber density features
4. Add to multimodal model as GNN layer

---

## 📊 REALISTIC DATA VOLUMES

| Source | Type | Est. Samples | Usable? |
|--------|------|--------------|---------|
| Cancelled Brands | Labeled text | 5-10k | ✅ YES |
| Cyberbullying IG/TikTok | Labeled text | 10k | ✅ YES |
| Jigsaw Toxic | Labeled text | 100k+ | ✅ YES |
| Hate Speech 1829 | Labeled text | 1.8k | ✅ YES |
| TweetBLM | Labeled text | 5k | ✅ YES |
| Sexism Balanced | Labeled text | 10k | ✅ YES |
| **Total Text** | | **~130k** | |
| MemeLens (all) | Labeled image-text | 20k | ✅ YES |
| Multilingual Violence | IDs only | 30k | ⚠️ Need text file |
| Pushshift 2023-01 | Unlabeled | 500k? | ⚠️ Only if you have 2024-2026 |
| Pushshift 2008-2012 | Unlabeled | Millions | ❌ NO |
| Raw Corpus | Unlabeled text | 400MB | ⚠️ DAPT only |

**Bottom Line**: You have **130k+ labeled text samples** and **20k labeled image-text pairs**. This is **MORE THAN ENOUGH** to train a production-ready model. Stop collecting, start training.

---

## 🎯 REVISED TIMELINE (Aggressive but Realistic)

| Week | Milestone | Deliverable | Viva Ready? |
|------|-----------|-------------|-------------|
| **1** | Baseline Model | RandomForest on 30k samples, 80% acc | Partial |
| **2** | DistilBERT Fine-tune | Text-only transformer, AUC > 0.88 | Yes (text demo) |
| **3** | Multimodal Fusion | CLIP + BERT, meme-aware doom scores | Yes (full demo) |
| **4** | Graph Integration | Neo4j populated, GNN features | Full |
| **5** | Attack Simulator | Adversarial examples generation | Full |
| **6** | Privacy Modules | DP/FL integration | Full |
| **7** | Polish & Demo | Streamlit app, viva rehearsal | **READY** |

**Total: 7 weeks** (not 6-9 months)

---

## 💀 HARD TRUTHS

1. **You don't have a data problem** - 130k labeled samples is a luxury
2. **You have an execution problem** - Infrastructure is world-class, models aren't trained
3. **Pushshift is a distraction** - Unless you prove 2024-2026 access, drop it
4. **CADE doesn't exist in your inventory** - Stop planning around it until you provide the path
5. **Multilingual violence dataset needs verification** - You claim text exists on HPC, prove it
6. **MemeLens is sufficient for multimodal** - 20k samples is enough to start

---

## 🚀 IMMEDIATE NEXT STEPS (Next 4 Hours)

### Step 1: Verify HPC Claims
```bash
# Run these on HPC and save output:
echo "=== Pushshift 2024-2026 ===" 
find /home/vivek.120542 -name "RC_202[4-6]*.zst" | wc -l

echo "=== Multilingual Violence Text ==="
find /home/vivek.120542/multilingual-twitter-collective-violence-dataset -name "*.csv" -o -name "*.txt" -o -name "*.parquet" | grep -v "geo"

echo "=== CADE Dataset ==="
find /home/vivek.120542 -iname "*cade*" 2>/dev/null
```

### Step 2: Copy Tier 1 Datasets to Workspace
```bash
mkdir -p /workspace/data/raw
cp "/home/vivek.120542/hate/Tweets on Cancelled Brands.csv" /workspace/data/raw/
cp /home/vivek.120542/cyberbullying-instagram-tiktok/train.parquet /workspace/data/raw/
cp /home/vivek.120542/hate/archive\ \(7\)/train.csv /workspace/data/raw/jigsaw_train.csv
cp /home/vivek.120542/hate/Social-Media-Toxic-Comments-Classification-main/data/train.csv /workspace/data/raw/toxic_train.csv
cp /home/vivek.120542/hate/hate_speech_1829.csv /workspace/data/raw/
cp /home/vivek.120542/hate/TweetBLM.csv /workspace/data/raw/
cp /home/vivek.120542/~/tum-nlp-sexism-socialmedia-balanced/sexism-socialmedia-balanced.csv /workspace/data/raw/
```

### Step 3: Run Consolidation Script
(I'll create this next)
```bash
python scripts/consolidate_datasets.py --data-dir /workspace/data/raw --output /workspace/data/unified_train.parquet
```

### Step 4: Train Baseline TODAY
```bash
python train_model_full_fixed.py --data-path /workspace/data/unified_train.parquet
```

---

## 📞 DECISION REQUIRED FROM YOU

Answer these questions before proceeding:

1. **Do you actually have 2024-2026 Pushshift files?** (Yes/No + count)
2. **Does the multilingual violence dataset have a text file on HPC?** (Yes/No + filename)
3. **Where exactly is the CADE dataset?** (Full path or admit it doesn't exist)
4. **Are you willing to abandon Pushshift 2008-2012 entirely?** (Yes/No)

Your answers will determine whether we:
- **Path A**: Proceed with existing 150k labeled samples (recommended)
- **Path B**: Waste weeks trying to validate/integrate questionable data

**Choose wisely.**
