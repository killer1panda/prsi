#!/usr/bin/env python3
"""
Production Dataset Consolidator for Doom Index v2
Consolidates CADE, Cyberbullying, Cancelled Brands, and Jigsaw into unified training set
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_cade(parquet_path: str) -> pd.DataFrame:
    """Load CADE dataset - contextual abuse detection"""
    print(f"Loading CADE from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # CADE schema: text, label (abusive/not_abusive), context_type
    required_cols = ['text', 'label']
    assert all(col in df.columns for col in required_cols), f"CADE missing columns: {df.columns}"
    
    # Normalize labels: abusive=1, not_abusive=0
    df['doom_score'] = df['label'].apply(lambda x: 1.0 if 'abusive' in str(x).lower() else 0.0)
    df['source'] = 'cade'
    df['modality'] = 'text'
    
    return df[['text', 'doom_score', 'source', 'modality']].dropna(subset=['text'])

def load_cyberbullying(parquet_dir: str) -> pd.DataFrame:
    """Load cyberbullying Instagram/TikTok dataset"""
    print(f"Loading Cyberbullying data from {parquet_dir}...")
    parquet_path = Path(parquet_dir)
    
    all_dfs = []
    for pq_file in parquet_path.glob('*.parquet'):
        df = pd.read_parquet(pq_file)
        all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Expected schema: text, label (bullying/not_bullying) or similar
    # Find text column
    text_col = None
    for col in ['text', 'comment', 'content', 'tweet']:
        if col in df.columns:
            text_col = col
            break
    
    label_col = None
    for col in ['label', 'category', 'severity', 'bullying']:
        if col in df.columns:
            label_col = col
            break
    
    assert text_col and label_col, f"Cyberbullying missing text/label cols. Found: {df.columns}"
    
    # Normalize to doom_score
    df['doom_score'] = df[label_col].apply(
        lambda x: 1.0 if any(term in str(x).lower() for term in ['bully', 'harass', 'severe', 'high']) else 0.0
    )
    df['source'] = 'cyberbullying'
    df['modality'] = 'text'
    
    return df[[text_col, 'doom_score', 'source', 'modality']].rename(columns={text_col: 'text'}).dropna(subset=['text'])

def load_cancelled_brands(csv_path: str) -> pd.DataFrame:
    """Load Tweets on Cancelled Brands - our gold standard"""
    print(f"Loading Cancelled Brands from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Expected: tweet_text, cancelled (yes/no) or similar
    text_col = None
    for col in ['tweet_text', 'text', 'tweet', 'content']:
        if col in df.columns:
            text_col = col
            break
    
    label_col = None
    for col in ['cancelled', 'label', 'status', 'outcome']:
        if col in df.columns:
            label_col = col
            break
    
    assert text_col and label_col, f"Cancelled Brands missing cols. Found: {df.columns}"
    
    # Normalize: cancelled=1, not_cancelled=0
    df['doom_score'] = df[label_col].apply(
        lambda x: 1.0 if any(term in str(x).lower() for term in ['yes', 'true', '1', 'cancelled']) else 0.0
    )
    df['source'] = 'cancelled_brands'
    df['modality'] = 'text'
    
    return df[[text_col, 'doom_score', 'source', 'modality']].rename(columns={text_col: 'text'}).dropna(subset=['text'])

def load_jigsaw(csv_path: str) -> pd.DataFrame:
    """Load Jigsaw Toxic Comments"""
    print(f"Loading Jigsaw from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Jigsaw schema: comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
    assert 'comment_text' in df.columns, f"Jigsaw missing comment_text. Found: {df.columns}"
    
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    available_tox_cols = [col for col in toxicity_cols if col in df.columns]
    
    assert len(available_tox_cols) > 0, "No toxicity columns found in Jigsaw"
    
    # Doom score = max of all toxicity indicators (if any are toxic, it's high doom)
    df['doom_score'] = df[available_tox_cols].max(axis=1)
    df['source'] = 'jigsaw'
    df['modality'] = 'text'
    
    return df[['comment_text', 'doom_score', 'source', 'modality']].rename(columns={'comment_text': 'text'}).dropna(subset=['text'])

def consolidate_datasets(data_dir: str, output_dir: str):
    """Main consolidation pipeline"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DOOM INDEX V2 - DATASET CONSOLIDATION")
    print("="*60)
    
    datasets = []
    
    # Load CADE
    cade_files = list(data_path.glob('*cade*.parquet')) + list((data_path / 'CADE_aequa' / 'data').glob('*.parquet'))
    if cade_files:
        datasets.append(load_cade(str(cade_files[0])))
    else:
        print("⚠️  CADE not found, skipping...")
    
    # Load Cyberbullying
    cyber_dirs = [data_path / 'cyberbullying-instagram-tiktok']
    for cdir in cyber_dirs:
        if cdir.exists():
            try:
                datasets.append(load_cyberbullying(str(cdir)))
                break
            except Exception as e:
                print(f"⚠️  Cyberbulloading failed: {e}")
    
    # Load Cancelled Brands
    cb_files = list(data_path.glob('*cancelled*.csv')) + list(data_path.glob('*brand*.csv'))
    if not cb_files:
        # Try exact name
        exact = data_path / 'Tweets on Cancelled Brands.csv'
        if exact.exists():
            cb_files = [exact]
    
    if cb_files:
        datasets.append(load_cancelled_brands(str(cb_files[0])))
    else:
        print("⚠️  Cancelled Brands not found, skipping...")
    
    # Load Jigsaw
    jigsaw_files = list(data_path.glob('*jigsaw*.csv'))
    if jigsaw_files:
        datasets.append(load_jigsaw(str(jigsaw_files[0])))
    else:
        print("⚠️  Jigsaw not found, skipping...")
    
    if not datasets:
        raise ValueError("No datasets loaded! Check your data_dir path.")
    
    # Concatenate all
    print(f"\n📊 Loaded {len(datasets)} datasets")
    combined = pd.concat(datasets, ignore_index=True)
    
    # Statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(combined):,}")
    print(f"Doom score range: [{combined['doom_score'].min():.2f}, {combined['doom_score'].max():.2f}]")
    print(f"Mean doom score: {combined['doom_score'].mean():.3f}")
    print(f"\nBy source:")
    print(combined['source'].value_counts())
    print(f"\nHigh-risk samples (doom > 0.5): {(combined['doom_score'] > 0.5).sum():,} ({(combined['doom_score'] > 0.5).mean()*100:.1f}%)")
    
    # Train/val/test split (stratified by doom_score buckets)
    combined['doom_bucket'] = pd.cut(combined['doom_score'], bins=[-0.01, 0.3, 0.7, 1.01], labels=['low', 'mid', 'high'])
    
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(
        combined, 
        test_size=0.3, 
        stratify=combined['doom_bucket'],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['doom_bucket'],
        random_state=42
    )
    
    # Drop bucket column
    for df in [train_df, val_df, test_df]:
        df.drop('doom_bucket', axis=1, inplace=True, errors='ignore')
    
    print(f"\n📈 Splits:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    train_df.to_parquet(output_path / f'train_{timestamp}.parquet', index=False)
    val_df.to_parquet(output_path / f'val_{timestamp}.parquet', index=False)
    test_df.to_parquet(output_path / f'test_{timestamp}.parquet', index=False)
    
    # Also save combined for inspection
    combined.to_parquet(output_path / f'full_consolidated_{timestamp}.parquet', index=False)
    
    print(f"\n✅ Saved to {output_path}")
    print(f"   - train_{timestamp}.parquet")
    print(f"   - val_{timestamp}.parquet")
    print(f"   - test_{timestamp}.parquet")
    print(f"   - full_consolidated_{timestamp}.parquet")
    
    return combined

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Consolidate Doom Index datasets')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Input data directory')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    args = parser.parse_args()
    
    consolidate_datasets(args.data_dir, args.output_dir)
