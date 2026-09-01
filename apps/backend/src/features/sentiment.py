"""Sentiment analysis features for doom-index.

This module provides sentiment analysis capabilities using:
- VADER (Valence Aware Dictionary and sEntiment Reasoner)
- HuggingFace Transformers (Mistral-7B-Instruct-based sentiment model)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not available")

try:
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, BitsAndBytesConfig, pipeline)

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available")


class SentimentAnalyzer:
    """Multi-backend sentiment analyzer with lazy loading."""

    def __init__(self):
        self.vader = None
        self._transformer_pipeline = None
        self._mistral_model = None
        self._mistral_tokenizer = None

        if VADER_AVAILABLE:
            try:
                self.vader = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.warning(f"VADER init error: {e}")

    @property
    def transformer_pipeline(self):
        """Lazy load Mistral pipeline on demand."""
        if self._transformer_pipeline is None and TRANSFORMERS_AVAILABLE:
            try:
                self._transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    return_all_scores=True,
                )
            except Exception as e:
                logger.warning(f"Could not load Mistral pipeline: {e}")
        return self._transformer_pipeline

    @property
    def mistral_model(self):
        """Lazy load Mistral model on demand with 4-bit quantization."""
        if self._mistral_model is None and TRANSFORMERS_AVAILABLE:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                self._mistral_tokenizer = AutoTokenizer.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.3"
                )
                self._mistral_tokenizer.pad_token = self._mistral_tokenizer.eos_token
                self._mistral_model = AutoModelForSequenceClassification.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            except Exception as e:
                logger.warning(f"Could not load Mistral model: {e}")
        return self._mistral_model

    @property
    def mistral_tokenizer(self):
        if self._mistral_tokenizer is None and TRANSFORMERS_AVAILABLE:
            _ = self.mistral_model
        return self._mistral_tokenizer

    def analyze(self, text: str) -> Dict[str, float]:
        """Convenience method returning sentiment scores (VADER fast path)."""
        vader_res = self.analyze_vader(text)
        if vader_res:
            return vader_res
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

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
            if not results:
                return None

            if isinstance(results, list):
                if isinstance(results[0], list):
                    seq = results[0]
                else:
                    seq = results
                scores = {}
                for result in seq:
                    if isinstance(result, dict) and "label" in result and "score" in result:
                        scores[result["label"]] = result["score"]
                return scores if scores else None
            return None
        except Exception as e:
            logger.error(f"Transformer analysis failed: {e}")
            return None

    def analyze_mistral(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze sentiment using Mistral-7B-Instruct for multimodal analysis.

        Uses last-token pooling (causal/decoder-only) then projects through
        the classification head.

        Returns:
            Dict with LABEL_0 (negative) and LABEL_1 (positive) scores
        """
        if not self.mistral_model or not self.mistral_tokenizer:
            return None

        try:
            inputs = self.mistral_tokenizer(
                text[:512], return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = self.mistral_model(**inputs, output_hidden_states=True)
                # Mistral is causal: take last token of last hidden state
                last_hidden = outputs.hidden_states[-1]  # [B, seq, hidden]
                seq_lengths = inputs["attention_mask"].sum(dim=-1) - 1  # [B]
                last_token = last_hidden[
                    torch.arange(last_hidden.size(0)), seq_lengths
                ]  # [B, hidden]
                # Project through classification head
                logits = self.mistral_model.score(last_token)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                scores = {
                    "LABEL_0": probabilities[0][0].item(),  # Negative
                    "LABEL_1": probabilities[0][1].item(),  # Positive
                }
            return scores
        except Exception as e:
            logger.error(f"Mistral analysis failed: {e}")
            return None

    def analyze_combined(self, text: str, include_transformers: bool = False) -> Dict[str, Any]:
        """Analyze sentiment using VADER with optional transformer enhancement.

        Returns:
            Dict containing vader, sentiment_compound, overall_sentiment, etc.
        """
        vader_res = self.analyze_vader(text) or {
            "compound": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "neu": 1.0,
        }
        compound = vader_res.get("compound", 0.0)

        result = {"vader": vader_res, "sentiment_compound": compound, "text_length": len(text)}

        if include_transformers:
            result["transformer"] = self.analyze_transformer(text)
            result["mistral"] = self.analyze_mistral(text)
        else:
            result["transformer"] = None
            result["mistral"] = None

        if compound >= 0.05:
            result["overall_sentiment"] = "positive"
        elif compound <= -0.05:
            result["overall_sentiment"] = "negative"
        else:
            result["overall_sentiment"] = "neutral"

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
