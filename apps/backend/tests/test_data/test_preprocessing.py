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
