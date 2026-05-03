import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/processed/unified_train.parquet')
    parser.add_argument('--output-dir', type=str, default='data/processed')
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Total samples: {len(df):,}")
    
    # First split: separate test set (10%)
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=(df['doom_score'] > 0.5).astype(int)
    )
    
    # Second split: separate val from train (10% of remaining = 9% total)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.111, random_state=42, 
        stratify=(train_val_df['doom_score'] > 0.5).astype(int)
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits with expected naming pattern
    train_path = output_dir / 'train_split.parquet'
    val_path = output_dir / 'val_split.parquet'
    test_path = output_dir / 'test_split.parquet'
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\n✅ Dataset split successfully:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"\nFiles saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

if __name__ == '__main__':
    main()
