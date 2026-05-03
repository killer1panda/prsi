import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

print("🔄 Restoring Full Cleaned Dataset...")

# Load original unified data
df = pd.read_parquet('data/processed/unified_train.parquet')
original_count = len(df)
print(f"Original loaded: {original_count:,}")

# 1. Filter Neutral Memes (doom_score == 0.5 exactly)
df_clean = df[df['doom_score'] != 0.5].copy()
print(f"After removing neutral memes: {len(df_clean):,}")

# 2. Filter Garbage Text (< 10 chars)
df_clean = df_clean[df_clean['text'].astype(str).str.len() > 10]
print(f"After length filter: {len(df_clean):,}")

# 3. Calculate Stats
positive_count = (df_clean['doom_score'] > 0.5).sum()
negative_count = (df_clean['doom_score'] <= 0.5).sum()
pos_ratio = positive_count / len(df_clean)

print(f"\n📊 Class Distribution:")
print(f"   Positive (>0.5): {positive_count:,} ({pos_ratio:.2%})")
print(f"   Negative (<=0.5): {negative_count:,} ({1-pos_ratio:.2%})")
print(f"   Total: {len(df_clean):,}")

# 4. Stratified Split (Keep natural imbalance)
train_df, temp_df = train_test_split(
    df_clean, test_size=0.2, random_state=42, 
    stratify=(df_clean['doom_score'] > 0.5).astype(int)
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42,
    stratify=(temp_df['doom_score'] > 0.5).astype(int)
)

# Save
out_dir = Path('data/processed')
out_dir.mkdir(parents=True, exist_ok=True)

train_df.to_parquet(out_dir / 'train_full.parquet', index=False)
val_df.to_parquet(out_dir / 'val_full.parquet', index=False)
test_df.to_parquet(out_dir / 'test_full.parquet', index=False)

print(f"\n✅ Saved Full Splits:")
print(f"   Train: {len(train_df):,} (Pos: {(train_df['doom_score']>0.5).sum():,})")
print(f"   Val:   {len(val_df):,} (Pos: {(val_df['doom_score']>0.5).sum():,})")
print(f"   Test:  {len(test_df):,} (Pos: {(test_df['doom_score']>0.5).sum():,})")
