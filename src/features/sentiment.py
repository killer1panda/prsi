"""Sentiment analysis features for doom-index.

This module provides sentiment analysis capabilities using:
- VADER (Valence Aware Dictionary and sEntiment Reasoner)
- HuggingFace Transformers (RoBERTa-based sentiment model)
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not available")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available")


class SentimentAnalyzer:
    """Multi-backend sentiment analyzer with DistilBERT."""

    def __init__(self):
        self.vader = None
        self.transformer_pipeline = None
        self.distilbert_model = None
        self.distilbert_tokenizer = None

        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()

        if TRANSFORMERS_AVAILABLE:
            try:
                # RoBERTa pipeline
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )

                # DistilBERT for multimodal analysis
                self.distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
                self.distilbert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

                logger.info("Loaded DistilBERT and RoBERTa models")
            except Exception as e:
                logger.warning(f"Could not load transformer models: {e}")

    def analyze_vader(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze sentiment using VADER.

        Returns:
            Dict with keys: neg, neu, pos, compound
        """
        if not self.vader:
            return None

        return self.vader.polarity_scores(text)

    def analyze_transformer(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze sentiment using HuggingFace transformer.

        Returns:
            Dict with keys: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
        """
        if not self.transformer_pipeline:
            return None

        try:
            results = self.transformer_pipeline(text[:512])  # Limit input length
            # The pipeline may return a list of dicts or a list of list of dicts
            if isinstance(results, list) and len(results) > 0:
                # If first element is a list, iterate over it
                if isinstance(results[0], list):
                    seq = results[0]
                else:
                    seq = results
                scores = {}
                for result in seq:
                    # Each result is a dict with 'label' and 'score'
                    scores[result['label']] = result['score']
                return scores
            else:
                return None
        except Exception as e:
            logger.error(f"Transformer analysis failed: {e}")
            return None

    def analyze_distilbert(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze sentiment using DistilBERT for multimodal analysis.

        Returns:
            Dict with LABEL_0 (negative) and LABEL_1 (positive) scores
        """
        if not self.distilbert_model or not self.distilbert_tokenizer:
            return None

        try:
            inputs = self.distilbert_tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.distilbert_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                scores = {
                    "LABEL_0": probabilities[0][0].item(),  # Negative
                    "LABEL_1": probabilities[0][1].item()   # Positive
                }
            return scores
        except Exception as e:
            logger.error(f"DistilBERT analysis failed: {e}")
            return None

    def analyze_combined(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using all available methods.

        Returns:
            Dict containing vader, transformer, and distilbert results
        """
        result = {
            'vader': self.analyze_vader(text),
            'transformer': self.analyze_transformer(text),
            'distilbert': self.analyze_distilbert(text),
            'text_length': len(text)
        }

        # Determine overall sentiment using multimodal approach
        if result['distilbert']:
            # Use DistilBERT as primary for multimodal analysis
            scores = result['distilbert']
            pos_score = scores.get('LABEL_1', 0)
            neg_score = scores.get('LABEL_0', 0)
            if pos_score > 0.6:
                result['overall_sentiment'] = 'positive'
            elif neg_score > 0.6:
                result['overall_sentiment'] = 'negative'
            else:
                result['overall_sentiment'] = 'neutral'
        elif result['vader']:
            # Fallback to VADER
            compound = result['vader']['compound']
            if compound >= 0.05:
                result['overall_sentiment'] = 'positive'
            elif compound <= -0.05:
                result['overall_sentiment'] = 'negative'
            else:
                result['overall_sentiment'] = 'neutral'
        elif result['transformer']:
            # Use transformer if others not available
            scores = result['transformer']
            if scores.get('LABEL_2', 0) > scores.get('LABEL_0', 0) and scores.get('LABEL_2', 0) > scores.get('LABEL_1', 0):
                result['overall_sentiment'] = 'positive'
            elif scores.get('LABEL_0', 0) > scores.get('LABEL_1', 0):
                result['overall_sentiment'] = 'negative'
            else:
                result['overall_sentiment'] = 'neutral'
        else:
            result['overall_sentiment'] = 'unknown'

        return result


# Global analyzer instance
_analyzer = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create global sentiment analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer

def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """Convenience function to analyze text sentiment."""
    analyzer = get_sentiment_analyzer()
    return analyzer.analyze_combined(text)