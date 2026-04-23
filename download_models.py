#!/usr/bin/env python3
"""Pre-download model weights for offline HPC usage.

Run this on a machine with internet access, then scp the cache to your HPC cluster.

Usage:
    python download_models.py
    scp -r ~/.cache/huggingface vivek.120542@10.16.1.50:~/.cache/
"""

import os
from pathlib import Path


def download_distilbert():
    """Download DistilBERT base uncased."""
    print("Downloading DistilBERT base uncased...")
    from transformers import DistilBertModel, DistilBertTokenizer

    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    print(f"  Model cached at: {Path.home()}/.cache/huggingface/")
    print(f"  Model size: ~250MB")


def download_sentiment_models():
    """Download sentiment analysis models."""
    print("Downloading sentiment models...")

    from transformers import pipeline

    # DistilBERT sentiment
    print("  - DistilBERT sentiment")
    pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # RoBERTa sentiment (if used)
    try:
        print("  - RoBERTa sentiment")
        pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    except Exception as e:
        print(f"    Skipped: {e}")


def download_toxicity_model():
    """Download toxicity classifier."""
    print("Downloading toxicity model...")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        model_name = "unitary/toxic-bert"
        AutoModelForSequenceClassification.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name)
        print(f"  - {model_name}")
    except Exception as e:
        print(f"  Skipped toxicity model: {e}")


def show_cache_info():
    """Show cache directory info."""
    cache_dir = Path.home() / ".cache" / "huggingface"

    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        total_size_mb = total_size / (1024 * 1024)

        print(f"\nCache directory: {cache_dir}")
        print(f"Total size: {total_size_mb:.1f} MB")
        print(f"\nTo copy to HPC:")
        print(f"  scp -r {cache_dir} username@hpc-host:~/.cache/")
    else:
        print(f"\nCache directory not found: {cache_dir}")


def main():
    print("=" * 60)
    print("Doom Index v2 — Model Weight Downloader")
    print("=" * 60)
    print()

    download_distilbert()
    print()

    download_sentiment_models()
    print()

    download_toxicity_model()
    print()

    show_cache_info()

    print()
    print("=" * 60)
    print("All models downloaded!")
    print("=" * 60)


if __name__ == "__main__":
    main()
