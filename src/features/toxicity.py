"""Toxicity detection features using Google Perspective API.

This module provides toxicity analysis for text content.
"""

import logging
from typing import Dict, Any, Optional
import requests
from src.config import get_env_var

logger = logging.getLogger(__name__)

class ToxicityAnalyzer:
    """Toxicity analyzer using Google Perspective API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_env_var("PERSPECTIVE_API_KEY")
        self.endpoint = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

        if not self.api_key:
            logger.warning("PERSPECTIVE_API_KEY not set. Toxicity analysis will be disabled.")

    def analyze(self, text: str) -> Dict[str, float]:
        """Convenience method returning toxicity scores."""
        return self.analyze_toxicity(text)

    def _local_heuristic(self, text: str) -> Dict[str, float]:
        """Local keyword-based heuristic fallback."""
        toxic_words = {'hate', 'stupid', 'cancel', 'terrible', 'horrible', 'idiot', 'disgusting', 'kill', 'evil', 'trash', 'scam', 'liar'}
        words = set(text.lower().split())
        matches = len(words & toxic_words)
        score = min(1.0, max(0.05, matches * 0.3))
        return {
            "toxicity": score,
            "toxicity_score": score,
            "severe_toxicity": score * 0.5,
            "identity_attack": 0.0,
            "insult": score * 0.8,
            "profanity": score * 0.4,
            "threat": 0.0
        }

    def analyze_toxicity(self, text: str) -> Dict[str, float]:
        """Analyze text for toxicity using Perspective API or local heuristic fallback."""
        if not self.api_key:
            return self._local_heuristic(text)

        data = {
            "comment": {"text": text[:3000]},  # API limit
            "languages": ["en"],
            "requestedAttributes": {
                "TOXICITY": {},
                "SEVERE_TOXICITY": {},
                "IDENTITY_ATTACK": {},
                "INSULT": {},
                "PROFANITY": {},
                "THREAT": {}
            }
        }

        try:
            response = requests.post(
                f"{self.endpoint}?key={self.api_key}",
                json=data,
                timeout=5
            )
            response.raise_for_status()
            result = response.json()

            scores = {}
            for attr, data_attr in result.get("attributeScores", {}).items():
                scores[attr.lower()] = data_attr.get("summaryScore", {}).get("value", 0.0)
            scores["toxicity_score"] = scores.get("toxicity", 0.0)

            return scores

        except Exception as e:
            logger.warning(f"Perspective API error ({e}). Using local heuristic fallback.")
            return self._local_heuristic(text)


    def is_toxic(self, text: str, threshold: float = 0.7) -> bool:
        """Check if text is toxic based on threshold."""
        scores = self.analyze_toxicity(text)
        if not scores:
            return False
        return scores.get("toxicity", 0) > threshold


# Global analyzer instance
_toxicity_analyzer = None

def get_toxicity_analyzer() -> ToxicityAnalyzer:
    """Get or create global toxicity analyzer instance."""
    global _toxicity_analyzer
    if _toxicity_analyzer is None:
        _toxicity_analyzer = ToxicityAnalyzer()
    return _toxicity_analyzer

def analyze_text_toxicity(text: str) -> Optional[Dict[str, float]]:
    """Convenience function to analyze text toxicity."""
    analyzer = get_toxicity_analyzer()
    return analyzer.analyze_toxicity(text)