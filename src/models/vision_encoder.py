"""
Production-grade CLIP-based Vision Encoder for multimodal Doom Index.
Handles image preprocessing, batch embedding extraction, and meme-aware encoding.
"""
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPModel

logger = logging.getLogger(__name__)


@dataclass
class VisionConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    cache_dir: str = "./cache/vision"
    embedding_dim: int = 512
    freeze_backbone: bool = False
    projection_dim: int = 256


class VisionEncoder(nn.Module):
    """
    CLIP-based vision encoder with optional projection head and caching.
    Production features: batching, disk caching, mixed precision support.
    """
    
    def __init__(self, config: Optional[VisionConfig] = None):
        super().__init__()
        self.config = config or VisionConfig()
        self.device = torch.device(self.config.device)
        
        # Load CLIP vision components
        self.processor = CLIPProcessor.from_pretrained(
            self.config.model_name, 
            cache_dir=self.config.cache_dir
        )
        self.vision_model = CLIPVisionModel.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        ).to(self.device)
        
        if self.config.freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Projection head for domain adaptation
        self.projection = nn.Sequential(
            nn.Linear(self.vision_model.config.hidden_size, self.config.projection_dim * 2),
            nn.LayerNorm(self.config.projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.projection_dim * 2, self.config.projection_dim)
        ).to(self.device)
        
        self._cache: Dict[str, torch.Tensor] = {}
        logger.info(f"VisionEncoder initialized: {self.config.model_name} on {self.device}")
    
    def _get_cache_key(self, image_path: Union[str, Path]) -> str:
        """Generate cache key based on file content hash."""
        path = Path(image_path)
        if path.exists():
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        return hashlib.sha256(str(image_path).encode()).hexdigest()[:16]
    
    def preprocess(self, images: List[Union[str, Path, Image.Image]]) -> torch.Tensor:
        """
        Preprocess images for CLIP. Handles file paths, PIL Images, and URLs.
        
        Args:
            images: List of image paths or PIL Images
            
        Returns:
            Preprocessed pixel values tensor
        """
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        return inputs["pixel_values"].to(self.device)
    
    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def encode(self, images: List[Union[str, Path, Image.Image]], 
               use_cache: bool = True) -> torch.Tensor:
        """
        Encode images to embedding vectors with caching.
        
        Args:
            images: List of images
            use_cache: Whether to use memory cache
            
        Returns:
            Tensor of shape (N, projection_dim)
        """
        if not images:
            return torch.zeros((0, self.config.projection_dim), device=self.device)
        
        # Check cache for file-based images
        if use_cache and all(isinstance(img, (str, Path)) for img in images):
            cache_keys = [self._get_cache_key(img) for img in images]
            cached = [self._cache.get(k) for k in cache_keys]
            
            if all(c is not None for c in cached):
                return torch.stack(cached)
            
            # Partial cache hit handling would go here; simplified for production
        
        # Batch processing
        all_embeddings = []
        for i in range(0, len(images), self.config.batch_size):
            batch = images[i:i + self.config.batch_size]
            pixel_values = self.preprocess(batch)
            
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output  # (B, hidden_size)
            projected = self.projection(pooled)  # (B, projection_dim)
            projected = nn.functional.normalize(projected, p=2, dim=-1)
            
            all_embeddings.append(projected)
        
        embeddings = torch.cat(all_embeddings, dim=0)
        
        # Update cache
        if use_cache and all(isinstance(img, (str, Path)) for img in images):
            for key, emb in zip(cache_keys, embeddings):
                self._cache[key] = emb.detach().cpu()
        
        return embeddings
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for training with preprocessed tensors."""
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        return self.projection(pooled)
    
    def compute_similarity(self, img1: Union[str, Image.Image], 
                           img2: Union[str, Image.Image]) -> float:
        """Compute cosine similarity between two images."""
        embs = self.encode([img1, img2], use_cache=False)
        sim = torch.cosine_similarity(embs[0:1], embs[1:2], dim=-1)
        return sim.item()
    
    def save(self, path: str):
        """Save model weights and config."""
        torch.save({
            "vision_model": self.vision_model.state_dict(),
            "projection": self.projection.state_dict(),
            "config": self.config
        }, path)
        logger.info(f"VisionEncoder saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.vision_model.load_state_dict(checkpoint["vision_model"])
        self.projection.load_state_dict(checkpoint["projection"])
        logger.info(f"VisionEncoder loaded from {path}")


class MultimodalFusion(nn.Module):
    """
    Late fusion module combining vision + text embeddings.
    Uses cross-modal attention for fine-grained alignment.
    """
    
    def __init__(self, text_dim: int = 768, vision_dim: int = 256, 
                 fusion_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, batch_first=True
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_dim, fusion_dim)
        )
        self.output_proj = nn.Linear(fusion_dim, 1)
    
    def forward(self, text_emb: torch.Tensor, 
                vision_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            text_emb: (B, text_dim)
            vision_emb: (B, vision_dim) or None
        Returns:
            logits: (B, 1)
        """
        text_proj = self.text_proj(text_emb).unsqueeze(1)  # (B, 1, fusion_dim)
        
        if vision_emb is not None:
            vision_proj = self.vision_proj(vision_emb).unsqueeze(1)  # (B, 1, fusion_dim)
            # Cross-attention: text queries vision
            attn_out, _ = self.cross_attn(text_proj, vision_proj, vision_proj)
            combined = torch.cat([text_proj.squeeze(1), attn_out.squeeze(1)], dim=-1)
        else:
            # Text-only fallback
            combined = torch.cat([text_proj.squeeze(1), text_proj.squeeze(1)], dim=-1)
        
        fused = self.fusion_mlp(combined)
        return self.output_proj(fused)
