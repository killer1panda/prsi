import os
import sys
from pathlib import Path

# Add backend to sys.path
backend_path = str(Path(__file__).resolve().parent / "apps" / "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import pandas as pd
from src.features import process_dataset_for_ml


def main():
    root_path = Path(__file__).resolve().parent
    candidate_paths = [
        root_path / "doom index" / "data" / "twitter_dataset" / "enhanced_tweets.csv",
        root_path / "processed_sample.csv",
    ]
    input_path = None
    for p in candidate_paths:
        if p.exists():
            input_path = str(p)
            break

    output_csv = str(root_path / "processed_sample.csv")

    if input_path:
        df = process_dataset_for_ml(
            input_path,
            output_csv=output_csv,
            sample_size=100
        )
        print(f"Processed sample shape: {df.shape}")
        print(df.head())
    else:
        print("No input CSV found; using synthetic sample...")
        dummy_df = pd.DataFrame({
            "text": ["I am outraged by this scandal!", "Everything is great today.", "Boycott this company now!"],
            "retweet_count": [10, 1, 45],
            "like_count": [20, 5, 90],
        })
        temp_path = str(root_path / "temp_input.csv")
        dummy_df.to_csv(temp_path, index=False)
        df = process_dataset_for_ml(temp_path, output_csv=output_csv, sample_size=3)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Processed synthetic sample shape: {df.shape}")


if __name__ == "__main__":
    main()