#!/usr/bin/env python3
"""Full model training script for HPC - processes Reddit NDJSON data."""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import hashlib
import re
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

def anonymize_user_id(user_id):
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:16]

# Keywords for filtering
keywords = [
    r'\bcancel\b', r'\bcancelled\b', r'\bcancellation\b', r'cancelled after',
    r'\bbacklash\b', r'\bcontroversy\b', r'under fire', r'\bboycott\b',
    r'\boutrage\b', r'\bpetition\b', r'facing backlash', r'called out'
]

def contains_keyword(text):
    if not text:
        return False
    text_lower = text.lower()
    for kw in keywords:
        if re.search(kw, text_lower, re.IGNORECASE):
            return True
    return False

def process_ndjson_to_df(ndjson_path, is_comment=True):
    """Process NDJSON file to DataFrame with cancellation-related posts."""
    data = []
    count = 0

    print(f"Processing {ndjson_path}...")

    with open(ndjson_path, 'r') as f:
        for line_num, line in enumerate(f):

            try:
                post = json.loads(line.strip())

                # Skip deleted authors
                if post.get('author') == '[deleted]':
                    continue

                # Get text content
                if is_comment:
                    text = post.get('body', '')
                else:
                    text = post.get('title', '') + ' ' + post.get('selftext', '')

                if not text.strip():
                    continue

                # Check for keywords
                if not contains_keyword(text):
                    continue

                # Convert timestamp
                created_at = datetime.fromtimestamp(int(post['created_utc'])).strftime('%a %b %d %H:%M:%S +0000 %Y')

                # Determine keyword
                primary_kw = 'cancelled'
                for kw in ['cancel', 'cancelled', 'backlash', 'controversy', 'boycott', 'outrage', 'petition']:
                    if kw in text.lower():
                        primary_kw = kw
                        break

                # Create row
                row = {
                    'id': post['id'],
                    'keyword': primary_kw,
                    'text': text.replace('\n', ' ').replace('\r', '')[:1000],
                    'created_at': created_at,
                    'author_id': anonymize_user_id(post['author']),
                    'user': post['author'],
                    'followers': 0,
                    'verified': False,
                    'likes': post.get('ups', post.get('score', 0)),
                    'retweets': 0,
                    'replies': 0,
                    'quotes': 0,
                    'hashtags': '',
                    'media_count': 0,
                    'media_urls': '[]',
                    'is_retweet': False,
                    'is_quote': False
                }

                data.append(row)
                count += 1

                if count % 1000 == 0:
                    print(f"Processed {count} relevant posts...")

            except Exception as e:
                continue

    print(f"Total relevant posts extracted: {len(data)}")
    return pd.DataFrame(data)

def add_engineered_features(df):
    """Add engineered features to DataFrame."""
    # Text features
    df['text_length'] = df['text'].fillna('').str.len()
    df['word_count'] = df['text'].fillna('').str.split().str.len()
    df['hashtag_count'] = df['hashtags'].fillna('').str.count('#')

    # Sentiment placeholders (no API)
    df['sentiment_vader'] = 'None'
    df['sentiment_transformer'] = 'None'
    df['toxicity_toxicity'] = 'None'

    # Sentiment features (defaults)
    df['sentiment_polarity'] = 0.0
    df['sentiment_intensity'] = 0.0
    df['toxicity_toxicity'] = 0.0
    df['is_toxic'] = 0

    # Time features
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['hour'] = df['created_at'].dt.hour.fillna(12)
    df['day_of_week'] = df['created_at'].dt.dayofweek.fillna(0)

    # Keyword features
    cancellation_keywords = ['cancel', 'cancelled', 'backlash', 'controversy', 'boycott', 'outrage', 'petition']
    for kw in cancellation_keywords:
        df[f'has_{kw}'] = df['text'].fillna('').str.lower().str.contains(kw).astype(int)

    return df

def create_feature_matrix(df):
    """Create feature matrix for ML."""
    feature_cols = [
        'likes', 'retweets', 'replies', 'quotes',
        'text_length', 'word_count', 'hashtag_count',
        'sentiment_polarity', 'sentiment_intensity',
        'toxicity_toxicity', 'is_toxic',
        'hour', 'day_of_week'
    ] + [f'has_{kw}' for kw in ['cancel', 'cancelled', 'backlash', 'controversy', 'boycott', 'outrage', 'petition']]

    # Filter available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].fillna(0).values

    # Target: has cancellation keywords
        # PROPER WEAK SUPERVISION LABELS (not just keyword matching)
    def create_proper_labels(df):
        scores = np.zeros(len(df))

        # High engagement = more visibility for backlash
        if 'likes' in df.columns:
            scores += (df['likes'] > df['likes'].quantile(0.9)).astype(int)

        # Very negative sentiment
        if 'sentiment_polarity' in df.columns:
            scores += (df['sentiment_polarity'] < -0.3).astype(int)

        # Strong action keywords (weighted higher)
        action_kw = ['boycott', 'petition', 'fired', 'removed', 'banned', 'apologized', 'resigned']
        has_action = df['text'].str.lower().apply(lambda t: any(kw in t for kw in action_kw))
        scores += has_action.astype(int) * 2

        # Controversy keywords
        controversy_kw = ['cancel', 'backlash', 'outrage', 'controversy', 'under fire']
        has_controversy = df['text'].str.lower().apply(lambda t: any(kw in t for kw in controversy_kw))
        scores += has_controversy.astype(int)

        # High reply ratio (engagement storm indicator)
        if 'replies' in df.columns and 'likes' in df.columns:
            reply_ratio = df['replies'] / (df['likes'] + 1)
            scores += (reply_ratio > reply_ratio.quantile(0.95)).astype(int)

        # Threshold: score >= 3 is a cancellation event
        return (scores >= 3).astype(int)

    y = create_proper_labels(df)

    return X, y

def main():
    """Main training function."""
    print("Starting Reddit data processing and model training...")

    # Process comments and submissions
    comments_df = process_ndjson_to_df('doom index/data/twitter_dataset/scraped_data/reddit/comments/RC_2008-12.ndjson', is_comment=True)
    submissions_df = process_ndjson_to_df('doom index/data/twitter_dataset/scraped_data/reddit/submissions/RS_2008-12.ndjson', is_comment=False)

    # Combine datasets
    df = pd.concat([comments_df, submissions_df], ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")

    # Add features
    df = add_engineered_features(df)

    # Create features and target
    X, y = create_feature_matrix(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\nTraining Results:")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(".3f")
    print(".3f")
    print(".3f")

    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/cancellation_predictor_full.pkl', 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': [col for col in df.columns if col in [
            'likes', 'retweets', 'replies', 'quotes', 'text_length', 'word_count', 'hashtag_count',
            'sentiment_polarity', 'sentiment_intensity', 'toxicity_toxicity', 'is_toxic', 'hour', 'day_of_week'
        ] + [f'has_{kw}' for kw in ['cancel', 'cancelled', 'backlash', 'controversy', 'boycott', 'outrage', 'petition']]]}, f)

    print("Training complete. Model saved to models/cancellation_predictor_full.pkl")

if __name__ == '__main__':
    main()