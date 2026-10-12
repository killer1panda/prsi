from src.features import process_dataset_for_ml

# Process full dataset
df = process_dataset_for_ml(
    '/Users/ajay/Downloads/doom-index/doom index/data/twitter_dataset/enhanced_tweets.csv',
    output_csv='/Users/ajay/Downloads/doom-index/processed_dataset.csv'
)

print(f"Full processed dataset shape: {df.shape}")
print("Ready for HPC training")