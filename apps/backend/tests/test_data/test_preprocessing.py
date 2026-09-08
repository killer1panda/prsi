"""Tests for data preprocessing module."""

import pytest
from src.data.preprocessing import DataPreprocessor, preprocess_posts


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""

    def test_clean_text_removes_urls(self):
        """Test URL removal from text."""
        preprocessor = DataPreprocessor()
        text = "Check out this link https://example.com and http://test.org"
        cleaned = preprocessor.clean_text(text, remove_urls=True)
        assert "https://example.com" not in cleaned
        assert "http://test.org" not in cleaned

    def test_clean_text_lowercase(self):
        """Test lowercase conversion."""
        preprocessor = DataPreprocessor()
        text = "HELLO World"
        cleaned = preprocessor.clean_text(text, lowercase=True)
        assert cleaned == "hello world"

    def test_clean_text_preserves_mentions(self):
        """Test mention preservation when remove_mentions=False."""
        preprocessor = DataPreprocessor()
        text = "Hello @user1 and @user2"
        cleaned = preprocessor.clean_text(text, remove_mentions=False)
        assert "@user1" in cleaned
        assert "@user2" in cleaned

    def test_anonymize_text_emails(self):
        """Test email anonymization."""
        preprocessor = DataPreprocessor()
        text = "Contact me at test@example.com"
        anonymized = preprocessor.anonymize_text(text)
        assert "test@example.com" not in anonymized
        assert "[EMAIL]" in anonymized

    def test_deduplicate_posts(self, sample_posts):
        """Test post deduplication."""
        preprocessor = DataPreprocessor()
        # Add duplicate
        posts_with_dup = sample_posts + [sample_posts[0]]
        unique = preprocessor.deduplicate_posts(posts_with_dup)
        assert len(unique) == len(sample_posts)

    def test_preprocess_pipeline(self, sample_posts):
        """Test full preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        processed = preprocessor.preprocess_pipeline(sample_posts)
        assert len(processed) == len(sample_posts)
        for post in processed:
            assert "cleaned_text" in post
            assert "preprocessed_at" in post

    def test_get_stats(self, sample_posts):
        """Test statistics retrieval."""
        preprocessor = DataPreprocessor()
        preprocessor.preprocess_pipeline(sample_posts)
        stats = preprocessor.get_stats()
        assert "processed" in stats
        assert stats["processed"] == len(sample_posts)

    def test_clean_text_preserves_emojis_by_default(self):
        """Test that emojis are preserved by default because they carry crucial emotion."""
        preprocessor = DataPreprocessor()
        text = "Breaking emergency 🚨 outraged users protesting 😡🔥"
        cleaned = preprocessor.clean_text(text)
        assert "🚨" in cleaned
        assert "😡" in cleaned
        assert "🔥" in cleaned

    def test_clean_text_demojize_emojis(self):
        """Test converting emojis to descriptive text tags."""
        preprocessor = DataPreprocessor()
        text = "Totally normal 🤡💀"
        demojized = preprocessor.clean_text(text, demojize_emojis=True)
        assert "clown" in demojized or "skull" in demojized

    def test_extract_emoji_features_outrage_and_panic(self):
        """Test emoji emotional feature extraction for outrage and panic."""
        preprocessor = DataPreprocessor()
        outrage_text = "Disaster in management 😡🤬🚨"
        features = preprocessor.extract_emoji_features(outrage_text)
        assert features["emoji_count"] == 3
        assert features["outrage_score"] > 0.6
        assert features["net_valence"] < 0.0

    def test_extract_emoji_features_irony_and_sarcasm(self):
        """Test emotional swinging and sarcasm detection from cynical emojis."""
        preprocessor = DataPreprocessor()
        sarcastic_text = "Great leadership team doing an amazing job 🤡💀"
        features = preprocessor.extract_emoji_features(sarcastic_text)
        assert features["irony_flag"] is True
        assert features["cynicism_score"] > 0.5
