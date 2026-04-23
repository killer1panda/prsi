#!/usr/bin/env python3
"""Fixed training data pipeline for Doom Index v2.

This script:
1. Processes ALL Pushshift NDJSON files (not just 2008-12)
2. Creates proper weak supervision labels (multivariate, not keyword matching)
3. Generates balanced dataset for multimodal training
4. Saves to data/processed_reddit_multimodal.csv

Usage:
    python train_model_full_fixed.py [--data_dir PATH] [--output PATH] [--max_files N]
"""

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.features import add_engineered_features, create_feature_matrix
from src.data.preprocessing import (
    remove_duplicates, filter_language, anonymize_data
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def process_ndjson_to_df(file_path: str, is_comment: bool = True) -> pd.DataFrame:
    """Process a single NDJSON file to DataFrame."""
    records = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                record = json.loads(line)

                if is_comment:
                    processed = {
                        'text': record.get('body', ''),
                        'author_id': record.get('author', ''),
                        'created_at': record.get('created_utc', 0),
                        'likes': record.get('score', 0),
                        'replies': record.get('num_comments', 0),
                        'subreddit': record.get('subreddit', ''),
                        'post_id': record.get('id', ''),
                    }
                else:
                    processed = {
                        'text': record.get('selftext', record.get('title', '')),
                        'author_id': record.get('author', ''),
                        'created_at': record.get('created_utc', 0),
                        'likes': record.get('score', 0),
                        'replies': record.get('num_comments', 0),
                        'subreddit': record.get('subreddit', ''),
                        'post_id': record.get('id', ''),
                    }

                if processed['text'] and len(processed['text']) > 10:
                    records.append(processed)

            except (json.JSONDecodeError, KeyError):
                continue

    df = pd.DataFrame(records)
    logger.info(f"Processed {file_path}: {len(df)} valid records")
    return df


def process_all_pushshift_files(data_dir: str, max_files: int = None) -> pd.DataFrame:
    """Process all Pushshift NDJSON files in directory."""
    data_path = Path(data_dir)

    # Find all NDJSON files
    comment_files = list(data_path.glob("**/comments/*.ndjson"))
    submission_files = list(data_path.glob("**/submissions/*.ndjson"))

    all_files = comment_files + submission_files

    if not all_files:
        logger.warning(f"No NDJSON files found in {data_dir}")
        # Try alternative paths
        alt_paths = [
            "doom index/data/twitter_dataset/scraped_data/reddit",
            "data/twitter_dataset/scraped_data/reddit",
            "data/reddit",
        ]
        for alt in alt_paths:
            if Path(alt).exists():
                all_files = list(Path(alt).glob("**/*.ndjson"))
                if all_files:
                    logger.info(f"Found files in alternative path: {alt}")
                    break

    if not all_files:
        raise FileNotFoundError(f"No Pushshift data found. Checked: {data_dir}")

    if max_files:
        all_files = all_files[:max_files]

    logger.info(f"Found {len(all_files)} NDJSON files to process")

    all_dfs = []
    for file_path in tqdm(all_files, desc="Processing Pushshift files"):
        is_comment = "comment" in str(file_path).lower()
        try:
            df = process_ndjson_to_df(str(file_path), is_comment=is_comment)
            if len(df) > 0:
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")

    if not all_dfs:
        raise ValueError("No valid dataframes produced from NDJSON files")

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} total records")

    return combined


def create_proper_labels(df: pd.DataFrame) -> pd.Series:
    """Create proper weak supervision labels for cancellation events.

    This is NOT keyword matching. It uses a multivariate heuristic
    that combines engagement, sentiment, action keywords, and reply patterns.
    """
    scores = np.zeros(len(df))

    # 1. High engagement = more visibility for backlash
    if 'likes' in df.columns:
        try:
            q90 = df['likes'].quantile(0.9)
            if q90 > 0:
                scores += (df['likes'] > q90).astype(int)
        except:
            pass

    # 2. Very negative sentiment
    if 'sentiment_polarity' in df.columns:
        scores += (df['sentiment_polarity'] < -0.3).astype(int)

    # 3. Strong action keywords (weighted higher - these indicate real consequences)
    action_kw = ['boycott', 'petition', 'fired', 'removed', 'banned', 
                 'apologized', 'resigned', 'stepped down', 'under investigation']
    has_action = df['text'].str.lower().apply(
        lambda t: any(kw in t for kw in action_kw)
    )
    scores += has_action.astype(int) * 2  # Weight = 2

    # 4. Controversy keywords (weighted lower - these are just signals)
    controversy_kw = ['cancel', 'backlash', 'outrage', 'controversy', 
                      'under fire', 'facing criticism', 'under pressure']
    has_controversy = df['text'].str.lower().apply(
        lambda t: any(kw in t for kw in controversy_kw)
    )
    scores += has_controversy.astype(int)

    # 5. High reply-to-like ratio (engagement storm indicator)
    if 'replies' in df.columns and 'likes' in df.columns:
        try:
            reply_ratio = df['replies'] / (df['likes'] + 1)
            q95 = reply_ratio.quantile(0.95)
            if q95 > 0:
                scores += (reply_ratio > q95).astype(int)
        except:
            pass

    # 6. Verified account + negative sentiment (high reach risk)
    if 'verified' in df.columns and 'sentiment_polarity' in df.columns:
        scores += ((df['verified'] == True) & (df['sentiment_polarity'] < -0.2)).astype(int)

    # Threshold: score >= 3 indicates a cancellation event
    labels = (scores >= 3).astype(int)

    logger.info(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    logger.info(f"Positive rate: {labels.mean():.2%}")

    return pd.Series(labels, index=df.index)


def balance_dataset(df: pd.DataFrame, target_ratio: float = 0.3, random_state: int = 42) -> pd.DataFrame:
    """Balance dataset by undersampling majority class.

    Args:
        df: DataFrame with 'label' column
        target_ratio: Target ratio of positive samples (0.3 = 30% positive)
    """
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]

    logger.info(f"Before balancing: {len(pos)} positive, {len(neg)} negative")

    if len(pos) == 0:
        logger.warning("No positive samples found! Check your labeling logic.")
        return df

    # Calculate how many negatives to keep
    target_neg = int(len(pos) * (1 - target_ratio) / target_ratio)

    if len(neg) > target_neg:
        neg = neg.sample(n=target_neg, random_state=random_state)

    balanced = pd.concat([pos, neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(f"After balancing: {len(balanced)} total ({balanced['label'].mean():.1%} positive)")

    return balanced


def main():
    parser = argparse.ArgumentParser(description="Process Pushshift data for Doom Index v2")
    parser.add_argument("--data_dir", type=str, default=".", 
                        help="Directory containing Pushshift NDJSON files")
    parser.add_argument("--output", type=str, default="data/processed_reddit_multimodal.csv",
                        help="Output CSV path")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of NDJSON files to process")
    parser.add_argument("--target_ratio", type=float, default=0.3,
                        help="Target positive sample ratio after balancing")
    parser.add_argument("--min_text_length", type=int, default=20,
                        help="Minimum text length to keep")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DOOM INDEX v2 - Data Processing Pipeline")
    logger.info("=" * 60)

    # 1. Process all Pushshift files
    logger.info("Step 1: Processing Pushshift NDJSON files...")
    df = process_all_pushshift_files(args.data_dir, max_files=args.max_files)

    # 2. Preprocessing
    logger.info("Step 2: Preprocessing...")
    df = remove_duplicates(df)
    df = filter_language(df)
    df = anonymize_data(df)

    # Filter short texts
    df = df[df['text'].str.len() >= args.min_text_length]
    logger.info(f"After filtering: {len(df)} records")

    # 3. Feature engineering
    logger.info("Step 3: Feature engineering...")
    df = add_engineered_features(df)

    # 4. Create proper labels
    logger.info("Step 4: Creating proper weak supervision labels...")
    df['label'] = create_proper_labels(df)

    # 5. Balance dataset
    logger.info("Step 5: Balancing dataset...")
    df = balance_dataset(df, target_ratio=args.target_ratio)

    # 6. Create feature matrix (for baseline comparison)
    logger.info("Step 6: Creating feature matrix...")
    X, y, feature_names = create_feature_matrix(df)
    logger.info(f"Feature matrix: {X.shape}, Features: {feature_names}")

    # 7. Save
    logger.info("Step 7: Saving processed data...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    # Also save a sample for quick testing
    sample = df.sample(n=min(1000, len(df)), random_state=42)
    sample_path = output_path.parent / "processed_sample.csv"
    sample.to_csv(sample_path, index=False)
    logger.info(f"Sample saved to {sample_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Positive samples: {df['label'].sum()} ({df['label'].mean():.1%})")
    logger.info(f"Unique authors: {df['author_id'].nunique()}")
    logger.info(f"Unique subreddits: {df['subreddit'].nunique()}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)
    logger.info("Next: Run train_multimodal.py to train the multimodal model")


if __name__ == "__main__":
    main()
