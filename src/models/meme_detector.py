"""
Meme detection and virality scoring using CLIP embeddings.
Detects known meme templates and estimates meme virality potential.
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
    template_dir: str = "data/meme_templates"
    similarity_threshold: float = 0.82
    virality_threshold: float = 0.65
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_k_templates: int = 3


class MemeDetector:
    """
    Detects memes by comparing against a bank of known templates.
    Also scores virality based on visual complexity and text overlay density.
    """
    
    def __init__(self, vision_encoder, config: Optional[MemeDetectorConfig] = None):
        self.vision_encoder = vision_encoder
        self.config = config or MemeDetectorConfig()
        self.device = torch.device(self.config.device)
        
        # Template embeddings bank
        self.template_embeddings: Dict[str, torch.Tensor] = {}
        self.template_metadata: Dict[str, Dict] = {}
        
        # Virality scoring MLP (trained on historical virality data)
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
                self.template_metadata[template_file.stem] = {
                    "name": template_file.stem,
                    "path": str(template_file),
                    "avg_virality": 0.7  # Placeholder; should be loaded from DB
                }
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")
    
    def detect(self, image: Union[str, Path, Image.Image]) -> Dict[str, any]:
        """
        Detect if image is a meme and identify template.
        
        Returns:
            Dict with keys: is_meme, template_matches, virality_score, 
                           confidence, meme_type
        """
        emb = self.vision_encoder.encode([image], use_cache=False)[0]
        
        if not self.template_embeddings:
            # No templates loaded; use heuristic virality score
            return {
                "is_meme": False,
                "template_matches": [],
                "virality_score": 0.0,
                "confidence": 0.0,
                "meme_type": "unknown"
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
        visual_features = self._extract_visual_features(image)
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
            "meme_type": sorted_sims[0][0] if is_meme else "original"
        }
    
    def _extract_visual_features(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """Extract heuristic visual features for virality prediction."""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        
        img_array = np.array(img)
        
        features = [
            img_array.std() / 255.0,  # Contrast
            np.mean(np.abs(np.diff(img_array, axis=0))) / 255.0,  # Vertical edge density
            np.mean(np.abs(np.diff(img_array, axis=1))) / 255.0,  # Horizontal edge density
            img.size[0] / img.size[1],  # Aspect ratio
            1.0 if img.size[0] < 500 else 0.0,  # Low resolution flag
            0.0, 0.0, 0.0, 0.0, 0.0  # Reserved for OCR text density, etc.
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
