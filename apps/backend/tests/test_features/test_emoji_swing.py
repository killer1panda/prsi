"""Tests for emoji emotional swinging and irony detection in sentiment analysis."""

import pytest
from src.features.sentiment import SentimentAnalyzer, analyze_text_sentiment


class TestEmojiEmotionalSwing:
    """Test suite verifying emojis act as primary emotional swing vectors."""

    def test_sarcasm_and_mockery_inverts_sentiment(self):
        """Test that cynical emojis (🤡, 💀) invert positive lexical polarity."""
        analyzer = SentimentAnalyzer()
        text = "Great leadership team doing an amazing job 🤡💀"
        result = analyzer.analyze_combined(text, incorporate_emoji_swing=True)

        assert result["emoji_metrics"]["irony_flag"] is True
        assert result["emoji_metrics"]["cynicism_score"] > 0.0
        assert result["sentiment_compound"] < 0.0  # Flipped from positive to negative

    def test_outrage_emojis_intensify_negative_sentiment(self):
        """Test that outrage emojis (😡, 🤬, 🔥, 🚨) amplify negative sentiment severity."""
        analyzer = SentimentAnalyzer()
        text_plain = "The system crashed during production"
        text_outraged = "The system crashed during production 😡🤬🔥🚨"

        res_plain = analyzer.analyze_combined(text_plain, incorporate_emoji_swing=True)
        res_outraged = analyzer.analyze_combined(text_outraged, incorporate_emoji_swing=True)

        assert res_outraged["emoji_metrics"]["outrage_score"] > 0.7
        assert res_outraged["sentiment_compound"] < res_plain["sentiment_compound"]

    def test_panic_emojis_amplify_distress(self):
        """Test that panic emojis (😱, 😭, 📉, 🆘) amplify distress score."""
        analyzer = SentimentAnalyzer()
        text = "Emergency alert everyone liquidate now 😱📉🆘"
        result = analyzer.analyze_combined(text, incorporate_emoji_swing=True)

        assert result["emoji_metrics"]["panic_score"] > 0.6
        assert result["sentiment_compound"] < -0.3

    def test_positive_emojis_amplify_enthusiasm(self):
        """Test that positive emojis (🚀, 🎉, ✨) boost positive sentiment."""
        analyzer = SentimentAnalyzer()
        text = "Just shipped the new release 🚀🎉✨"
        result = analyzer.analyze_combined(text, incorporate_emoji_swing=True)

        assert result["emoji_metrics"]["positive_score"] > 0.7
        assert result["sentiment_compound"] > 0.5

    def test_neutral_text_unaltered(self):
        """Test that text without emojis maintains standard sentiment score."""
        analyzer = SentimentAnalyzer()
        text = "The meeting is scheduled for tomorrow afternoon."
        result = analyzer.analyze_combined(text, incorporate_emoji_swing=True)

        assert result["emoji_metrics"]["emoji_count"] == 0
        assert result["sentiment_compound"] == result["base_compound"]
