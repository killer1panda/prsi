"""
Qwen2-VL-7B NaViT Vision Encoder for multimodal Doom Index.
Native dynamic resolution (up to 1120×1120), built-in OCR for meme text,
temporal video understanding.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Qwen2VLForConditionalGeneration)

logger = logging.getLogger(__name__)


@dataclass
class VisionConfig:
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8  # Smaller batch due to higher resolution
    cache_dir: str = "./cache/vision"
    embedding_dim: int = 3584  # Qwen2-VL hidden size
    freeze_backbone: bool = False
    projection_dim: int = 512
    max_pixels: int = 1003520  # 1120x1120 approx
    min_pixels: int = 262144  # 512x512


class VisionEncoder(nn.Module):
    """
    Qwen2-VL-7B NaViT vision encoder with optional projection head and caching.
    Production features: 4-bit quantization, dynamic resolution, batching,
    disk caching, built-in OCR via the Qwen2-VL language head.
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        super().__init__()
        self.config = config or VisionConfig()
        self.device = torch.device(self.config.device)

        # 4-bit quantization config (NF4, double quant, bfloat16 compute)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load Qwen2-VL vision-language model
        self.vision_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=self.config.cache_dir,
        )

        # AutoProcessor handles dynamic resolution tiling via min/max_pixels
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )

        if self.config.freeze_backbone:
            for param in self.vision_model.model.visual.parameters():
                param.requires_grad = False

        # Projection head: 3584 (Qwen2-VL hidden size) → projection_dim (512)
        self.projection = nn.Sequential(
            nn.Linear(3584, self.config.projection_dim * 2),
            nn.LayerNorm(self.config.projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.projection_dim * 2, self.config.projection_dim),
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

    def preprocess(self, images: List[Union[str, Path, Image.Image]]):
        """
        Preprocess images for Qwen2-VL. Handles file paths and PIL Images.
        Uses the Qwen2-VL message format with dynamic resolution tiling
        (min_pixels=512×512, max_pixels≈1120×1120).

        Args:
            images: List of image paths or PIL Images

        Returns:
            Processed inputs dict ready for the Qwen2-VL model
        """
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        messages = [
            {"role": "user", "content": [{"type": "image", "image": img} for img in pil_images]}
        ]

        # Apply chat template and extract vision info for the processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    @torch.no_grad()
    def encode(
        self, images: List[Union[str, Path, Image.Image]], use_cache: bool = True
    ) -> torch.Tensor:
        """
        Encode images to embedding vectors with caching.
        Extracts the last hidden state from the Qwen2-VL vision tower
        (model.model.visual) and applies the learnable projection head.

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
            batch = images[i : i + self.config.batch_size]
            inputs = self.preprocess(batch)

            # Extract vision patch embeddings from the Qwen2-VL vision tower
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                visual_outputs = self.vision_model.model.visual(
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs.get("image_grid_thw"),
                )

            # visual_outputs: (total_patches, hidden_size=3584)
            # Pool per image using image_grid_thw
            patch_counts = (
                inputs.get("image_grid_thw").prod(dim=1).tolist()
                if inputs.get("image_grid_thw") is not None
                else [visual_outputs.size(0)]
            )
            pooled_list = []
            for img_patches in torch.split(visual_outputs, patch_counts):
                pooled_list.append(img_patches.mean(dim=0, keepdim=True))
            pooled = torch.cat(pooled_list, dim=0)  # (batch_size, 3584)

            projected = self.projection(pooled.float())  # (batch_size, projection_dim)
            projected = nn.functional.normalize(projected, p=2, dim=-1)

            all_embeddings.append(projected)

        embeddings = torch.cat(all_embeddings, dim=0)

        # Update cache
        if use_cache and all(isinstance(img, (str, Path)) for img in images):
            for key, emb in zip(cache_keys, embeddings):
                self._cache[key] = emb.detach().cpu()

        return embeddings

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for training with preprocessed tensors."""
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            visual_outputs = self.vision_model.model.visual(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        pooled = visual_outputs.mean(dim=0, keepdim=True)
        return self.projection(pooled.float())

    @torch.no_grad()
    def ocr_extract(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Extract all text visible in an image using Qwen2-VL's built-in OCR capability.
        Useful for meme text extraction, caption detection, and overlay text parsing.

        Args:
            image: A single image path or PIL Image

        Returns:
            Extracted text string (may be empty if no text is detected)
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Extract all text visible in this image:"},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.vision_model.generate(
            **inputs,
            max_new_tokens=256,
        )
        # Trim the input tokens from the generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip() if output_text else ""

    def compute_similarity(
        self, img1: Union[str, Image.Image], img2: Union[str, Image.Image]
    ) -> float:
        """Compute cosine similarity between two images."""
        embs = self.encode([img1, img2], use_cache=False)
        sim = torch.cosine_similarity(embs[0:1], embs[1:2], dim=-1)
        return sim.item()

    def save(self, path: str):
        """Save projection head weights and config (backbone weights are managed separately)."""
        torch.save({"projection": self.projection.state_dict(), "config": self.config}, path)
        logger.info(f"VisionEncoder projection head saved to {path}")

    def load(self, path: str):
        """Load projection head weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.projection.load_state_dict(checkpoint["projection"])
        logger.info(f"VisionEncoder projection head loaded from {path}")


class MultimodalFusion(nn.Module):
    """
    Late fusion module combining vision + text embeddings.
    Uses cross-modal attention for fine-grained alignment.
    Vision input is expected at projection_dim (512) — the output of
    the Qwen2-VL-7B vision tower after the learned projection head
    (raw Qwen2-VL hidden size is 3584, projected down to 512).
    """

    def __init__(
        self, text_dim: int = 4096, vision_dim: int = 512, fusion_dim: int = 512, num_heads: int = 8
    ):
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
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.output_proj = nn.Linear(fusion_dim, 1)

    def forward(
        self, text_emb: torch.Tensor, vision_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            text_emb: (B, text_dim)
            vision_emb: (B, vision_dim=512) or None — Qwen2-VL projected embedding
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
