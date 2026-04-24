"""
Multilingual text encoder for Indian social media (Hinglish, Hindi, English).
Uses XLM-RoBERTa with domain adaptation for code-switched text.
"""
import logging
import re
from typing import List, Optional, Dict, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


@dataclass
class MultilingualConfig:
    model_name: str = "xlm-roberta-base"  # or "ai4bharat/indic-bert"
    max_length: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_layers: int = 8  # Freeze first N transformer layers
    projection_dim: int = 256
    dropout: float = 0.15


class MultilingualEncoder(nn.Module):
    """
    Production multilingual encoder with language detection and 
    code-switching handling for Indian social media.
    """

    def __init__(self, config: Optional[MultilingualConfig] = None):
        super().__init__()
        self.config = config or MultilingualConfig()
        self.device = torch.device(self.config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.backbone = AutoModel.from_pretrained(self.config.model_name).to(self.device)

        # Freeze lower layers for transfer learning stability
        if self.config.freeze_layers > 0:
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False
            for layer in self.backbone.encoder.layer[:self.config.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_size // 2, self.config.projection_dim)
        ).to(self.device)

        # Language detection head (auxiliary task)
        self.lang_classifier = nn.Linear(hidden_size, 3).to(self.device)  # en, hi, mixed

        # Hinglish normalization patterns
        self.hinglish_patterns = self._compile_hinglish_patterns()

        logger.info(f"MultilingualEncoder loaded: {self.config.model_name}")

    def _compile_hinglish_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for common Hinglish constructs."""
        return {
            "roman_hindi": re.compile(r'\b(kya|nahi|hai|main|tu|aap|kaise|kyun|bahut|achha|bura)\b', re.I),
            "hindi_script": re.compile(r'[\u0900-\u097F]+'),
            "english": re.compile(r'\b(the|is|are|was|were|have|has|had|do|does|did|will|would|could|should)\b', re.I),
            "emoji_heavy": re.compile(r'[\U0001F600-\U0001F64F]{3,}'),
        }

    def detect_language(self, text: str) -> str:
        """
        Detect language mix: 'en', 'hi', 'hinglish', or 'mixed'.
        """
        has_hindi = bool(self.hinglish_patterns["hindi_script"].search(text))
        has_roman_hindi = bool(self.hinglish_patterns["roman_hindi"].search(text))
        has_english = bool(self.hinglish_patterns["english"].search(text))

        if has_hindi and has_english:
            return "mixed"
        elif has_hindi:
            return "hi"
        elif has_roman_hindi and has_english:
            return "hinglish"
        elif has_roman_hindi:
            return "hinglish"
        else:
            return "en"

    def preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess Hinglish and code-switched text.
        Normalizes common Roman Hindi spellings.
        """
        normalized = []
        for text in texts:
            # Normalize repeated characters (e.g., "haaaa" -> "haa")
            text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
            # Normalize spaces around punctuation
            text = re.sub(r'\s+([.,!?])', r'\1', text)
            normalized.append(text.lower().strip())
        return normalized

    def encode(self, texts: List[str], return_lang_logits: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode texts to embeddings.

        Args:
            texts: List of raw texts
            return_lang_logits: If True, also return language classification logits
        Returns:
            embeddings: (B, projection_dim)
            lang_logits: (B, 3) if return_lang_logits=True
        """
        texts = self.preprocess(texts)

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.backbone(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token

        embeddings = self.projection(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        if return_lang_logits:
            lang_logits = self.lang_classifier(pooled)
            return embeddings, lang_logits

        return embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.projection(pooled)

    def save(self, path: str):
        torch.save({
            "backbone": self.backbone.state_dict(),
            "projection": self.projection.state_dict(),
            "lang_classifier": self.lang_classifier.state_dict(),
            "config": self.config
        }, path)
        logger.info(f"MultilingualEncoder saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint["backbone"])
        self.projection.load_state_dict(checkpoint["projection"])
        self.lang_classifier.load_state_dict(checkpoint["lang_classifier"])
        logger.info(f"MultilingualEncoder loaded from {path}")
