import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("🧹 Cleaning Dataset...")

# Load unified data
df = pd.read_parquet('data/processed/unified_train.parquet')
print(f"Original: {len(df):,} samples")

# Filter: Remove low-signal meme-only samples (doom exactly 0.5)
# Keep only samples with clear positive/negative signals
df_clean = df[
    (df['doom_score'] < 0.45) |  # Clearly negative/safe
    (df['doom_score'] > 0.55)    # Clearly positive/doomed
].copy()

print(f"After filtering neutral memes: {len(df_clean):,} samples")

# Optional: Filter by text length (remove empty/too short)
df_clean = df_clean[df_clean['text'].str.len() > 10].copy()
print(f"After length filter: {len(df_clean):,} samples")

# Check class balance
positive = (df_clean['doom_score'] > 0.5).sum()
negative = len(df_clean) - positive
print(f"Positive: {positive:,} ({positive/len(df_clean)*100:.1f}%)")
print(f"Negative: {negative:,} ({negative/len(df_clean)*100:.1f}%)")

# If heavily imbalanced, downsample majority class
if positive / len(df_clean) < 0.3 or positive / len(df_clean) > 0.7:
    print("⚖️  Rebalancing dataset...")
    df_pos = df_clean[df_clean['doom_score'] > 0.5]
    df_neg = df_clean[df_clean['doom_score'] <= 0.5]
    
    # Downsample larger class to match smaller
    n_min = min(len(df_pos), len(df_neg))
    df_balanced = pd.concat([
        df_pos.sample(n=n_min, random_state=42),
        df_neg.sample(n=n_min, random_state=42)
    ]).sample(frac=1, random_state=42)
    
    print(f"Balanced dataset: {len(df_balanced):,} samples (50/50 split)")
else:
    df_balanced = df_clean
    print("✅ Dataset already balanced enough")

# Split 80/10/10
train_df, temp_df = train_test_split(
    df_balanced, test_size=0.2, random_state=42, 
    stratify=(df_balanced['doom_score'] > 0.5).astype(int)
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42,
    stratify=(temp_df['doom_score'] > 0.5).astype(int)
)

# Save cleaned splits
train_df.to_parquet('data/processed/clean_train.parquet', index=False)
val_df.to_parquet('data/processed/clean_val.parquet', index=False)
test_df.to_parquet('data/processed/clean_test.parquet', index=False)

print(f"\n✅ Saved:")
print(f"  Train: {len(train_df):,}")
print(f"  Val:   {len(val_df):,}")
print(f"  Test:  {len(test_df):,}")
