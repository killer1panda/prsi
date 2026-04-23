import pandas as pd
from src.features import process_dataset_for_ml

# Process a small sample
df = process_dataset_for_ml(
    '/Users/ajay/Downloads/doom-index/doom index/data/twitter_dataset/enhanced_tweets.csv',
    output_csv='/Users/ajay/Downloads/doom-index/processed_sample.csv',
    sample_size=100
)

print(f"Processed sample shape: {df.shape}")
print(df.head())