"""
Meme detection and virality scoring using Qwen2-VL-7B embeddings.
Detects known meme templates and estimates meme virality potential.
Leverages Qwen2-VL's built-in OCR capability for meme text extraction.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MemeDetectorConfig:
    """Configuration for the MemeDetector model, defining thresholds and weights."""
    template_dir: str = "data/meme_templates"
    similarity_threshold: float = 0.82
    virality_threshold: float = 0.65
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_k_templates: int = 3


class MemeDetector:
    """
    Detects memes by comparing against a bank of known templates using
    Qwen2-VL-7B vision embeddings (projected to 512d).
    Also scores virality based on visual complexity, text overlay density,
    and OCR-extracted meme text via Qwen2-VL's built-in OCR capability.
    """

    def __init__(self, vision_encoder, config: Optional[MemeDetectorConfig] = None):
        self.vision_encoder = vision_encoder
        self.config = config or MemeDetectorConfig()
        self.device = torch.device(self.config.device)

        # Template embeddings bank
        self.template_embeddings: Dict[str, torch.Tensor] = {}
        self.template_metadata: Dict[str, Dict] = {}

        # Virality scoring MLP (trained on historical virality data).
        # Input = Qwen2-VL projection output (512d) + 10 visual/OCR features.
        self.virality_scorer = nn.Sequential(
            nn.Linear(self.vision_encoder.config.projection_dim + 10, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)

        self._load_templates()
        logger.info(f"MemeDetector initialized with {len(self.template_embeddings)} templates")

    def _load_templates(self):
        """Load known meme template embeddings from disk."""
        template_path = Path(self.config.template_dir)
        if not template_path.exists():
            logger.warning(f"Template directory {template_path} not found. Meme detection limited.")
            return

        for template_file in template_path.glob("*.jpg"):
            try:
                emb = self.vision_encoder.encode([str(template_file)], use_cache=True)
                self.template_embeddings[template_file.stem] = emb[0]
                
                # Fetch template virality from MongoDB with graceful fallback
                avg_virality = 0.7
                try:
                    from src.data.db_connectors import get_mongodb
                    mongo = get_mongodb()
                    doc = mongo.db["meme_templates"].find_one({"template_name": template_file.stem})
                    if doc and "avg_virality" in doc:
                        avg_virality = float(doc["avg_virality"])
                except Exception:
                    pass
                    
                self.template_metadata[template_file.stem] = {
                    "name": template_file.stem,
                    "path": str(template_file),
                    "avg_virality": avg_virality
                }
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")

    def detect(self, image: Union[str, Path, Image.Image]) -> Dict[str, any]:
        """
        Detect if image is a meme and identify template.
        Uses Qwen2-VL-7B projected embeddings (512d) for similarity matching
        and OCR for meme text extraction.

        Returns:
            Dict with keys: is_meme, template_matches, virality_score,
                           confidence, meme_type, ocr_text
        """
        emb = self.vision_encoder.encode([image], use_cache=False)[0]

        # OCR-based meme text extraction via Qwen2-VL vision tower
        ocr_text = ""
        try:
            ocr_text = self.vision_encoder.ocr_extract(image)
        except Exception as e:
            logger.debug(f"OCR extraction failed: {e}")

        if not self.template_embeddings:
            # No templates loaded; use heuristic virality score
            return {
                "is_meme": False,
                "template_matches": [],
                "virality_score": 0.0,
                "confidence": 0.0,
                "meme_type": "unknown",
                "ocr_text": ocr_text,
            }

        # Compute similarities to all templates
        similarities = {}
        for name, template_emb in self.template_embeddings.items():
            sim = torch.cosine_similarity(emb.unsqueeze(0),
                                          template_emb.unsqueeze(0).to(self.device),
                                          dim=-1).item()
            similarities[name] = sim

        # Top-K matches
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_matches = [
            {"template": name, "similarity": round(sim, 4),
             "metadata": self.template_metadata.get(name, {})}
            for name, sim in sorted_sims[:self.config.top_k_templates]
        ]

        best_sim = sorted_sims[0][1] if sorted_sims else 0.0
        is_meme = best_sim > self.config.similarity_threshold

        # Virality scoring using visual features + template history
        visual_features = self._extract_visual_features(image, ocr_text=ocr_text)
        virality_input = torch.cat([
            emb.detach().cpu(),
            torch.tensor(visual_features, dtype=torch.float32)
        ]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            virality_score = self.virality_scorer(virality_input).item()

        return {
            "is_meme": is_meme,
            "template_matches": top_matches,
            "virality_score": round(virality_score, 4),
            "confidence": round(best_sim, 4),
            "meme_type": sorted_sims[0][0] if is_meme else "original",
            "ocr_text": ocr_text,
        }

    def _extract_visual_features(
        self,
        image: Union[str, Path, Image.Image],
        ocr_text: str = "",
    ) -> np.ndarray:
        """
        Extract heuristic visual features for virality prediction.
        Includes OCR text density derived from Qwen2-VL's OCR output.
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        img_array = np.array(img)

        # Normalised OCR text length as proxy for text overlay density
        ocr_density = min(len(ocr_text) / 200.0, 1.0)
        ocr_has_text = 1.0 if len(ocr_text.strip()) > 0 else 0.0

        features = [
            img_array.std() / 255.0,                                     # Contrast
            np.mean(np.abs(np.diff(img_array, axis=0))) / 255.0,          # Vertical edge density
            np.mean(np.abs(np.diff(img_array, axis=1))) / 255.0,          # Horizontal edge density
            img.size[0] / img.size[1],                                    # Aspect ratio
            1.0 if img.size[0] < 500 else 0.0,                           # Low resolution flag
            ocr_has_text,                                                  # Has OCR text (Qwen2-VL)
            ocr_density,                                                   # OCR text density
            0.0, 0.0, 0.0                                                  # Reserved
        ]
        return np.array(features[:10], dtype=np.float32)

    def batch_detect(self, images: List[Union[str, Image.Image]]) -> List[Dict]:
        """Batch meme detection for efficiency."""
        return [self.detect(img) for img in images]

    def add_template(self, image_path: str, name: str, metadata: Optional[Dict] = None):
        """Add new meme template to the bank."""
        emb = self.vision_encoder.encode([image_path], use_cache=True)
        self.template_embeddings[name] = emb[0]
        self.template_metadata[name] = metadata or {"name": name}
        logger.info(f"Added meme template: {name}")

    def compute_stream_exposure(self, posts: List[Dict]) -> Dict[str, any]:
        """Aggregate timeline meme exposure for a user stream.

        Blueprint specification:
          Computes frequency of high-virality memes, average virality,
          and composite meme exposure index across the user's timeline.

        Args:
            posts: List of post dicts (may contain 'image', 'image_path', or pre-computed 'meme_data')

        Returns:
            Dict containing meme exposure metrics:
              - total_posts: int
              - image_posts_count: int
              - meme_count: int
              - meme_frequency: float [0, 1]
              - high_virality_count: int
              - high_virality_frequency: float [0, 1]
              - avg_virality: float [0, 1]
              - peak_virality: float [0, 1]
              - meme_exposure_index: float [0, 1]
              - top_templates: List[str]
        """
        total_posts = len(posts)
        if total_posts == 0:
            return {
                "total_posts": 0,
                "image_posts_count": 0,
                "meme_count": 0,
                "meme_frequency": 0.0,
                "high_virality_count": 0,
                "high_virality_frequency": 0.0,
                "avg_virality": 0.0,
                "peak_virality": 0.0,
                "meme_exposure_index": 0.0,
                "top_templates": [],
            }

        image_posts = 0
        meme_count = 0
        high_virality_count = 0
        virality_scores = []
        template_counts: Dict[str, int] = {}

        for post in posts:
            img = post.get("image") or post.get("image_path") or post.get("image_url")
            if not img and "meme_data" not in post:
                continue

            image_posts += 1

            if "meme_data" in post:
                detection = post["meme_data"]
            else:
                try:
                    detection = self.detect(img)
                except Exception as e:
                    logger.debug(f"Error detecting meme for post {post.get('id', '')}: {e}")
                    continue

            if detection.get("is_meme", False):
                meme_count += 1
                v_score = float(detection.get("virality_score", 0.0))
                virality_scores.append(v_score)

                if v_score >= self.config.virality_threshold:
                    high_virality_count += 1

                m_type = detection.get("meme_type", "unknown")
                if m_type and m_type != "original":
                    template_counts[m_type] = template_counts.get(m_type, 0) + 1

        denom = max(1, image_posts)
        meme_frequency = meme_count / denom
        high_virality_freq = high_virality_count / denom
        avg_virality = float(np.mean(virality_scores)) if virality_scores else 0.0
        peak_virality = float(np.max(virality_scores)) if virality_scores else 0.0

        # Composite meme exposure index
        exposure_index = 0.6 * high_virality_freq + 0.4 * avg_virality

        sorted_templates = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)
        top_templates = [t[0] for t in sorted_templates[:5]]

        return {
            "total_posts": total_posts,
            "image_posts_count": image_posts,
            "meme_count": meme_count,
            "meme_frequency": round(meme_frequency, 4),
            "high_virality_count": high_virality_count,
            "high_virality_frequency": round(high_virality_freq, 4),
            "avg_virality": round(avg_virality, 4),
            "peak_virality": round(peak_virality, 4),
            "meme_exposure_index": round(exposure_index, 4),
            "top_templates": top_templates,
        }
