import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/processed/unified_train.parquet')
    parser.add_argument('--output-dir', type=str, default='data/processed')
    args = parser.parse_args()
    
    print("🧹 Cleaning Dataset...")
    df = pd.read_parquet(args.input)
    print(f"Original: {len(df):,} samples")
    
    # 1. Filter out neutral memes (doom_score == 0.5 exactly)
    df_filtered = df[df['doom_score'] != 0.5].copy()
    print(f"After filtering neutral memes: {len(df_filtered):,} samples")
    
    # 2. Remove very short texts (< 10 chars)
    df_filtered = df_filtered[df_filtered['text'].str.len() >= 10]
    print(f"After length filter: {len(df_filtered):,} samples")
    
    # 3. Calculate class distribution
    positive_count = (df_filtered['doom_score'] > 0.5).sum()
    negative_count = (df_filtered['doom_score'] <= 0.5).sum()
    print(f"Positive: {positive_count:,} ({positive_count/len(df_filtered)*100:.1f}%)")
    print(f"Negative: {negative_count:,} ({negative_count/len(df_filtered)*100:.1f}%)")
    
    # 4. Split data (preserve natural imbalance)
    train_val_df, test_df = train_test_split(
        df_filtered, 
        test_size=0.1, 
        random_state=42, 
        stratify=(df_filtered['doom_score'] > 0.5).astype(int)
    )
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.111, 
        random_state=42,
        stratify=(train_val_df['doom_score'] > 0.5).astype(int)
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with clear naming
    train_path = output_dir / 'cleaned_train.parquet'
    val_path = output_dir / 'cleaned_val.parquet'
    test_path = output_dir / 'cleaned_test.parquet'
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\n✅ Saved cleaned splits:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")
    print(f"\nFiles: {train_path}, {val_path}, {test_path}")

if __name__ == '__main__':
    main()
