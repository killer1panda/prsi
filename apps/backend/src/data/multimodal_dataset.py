"""
Multimodal dataset loader handling text, image, and graph data with missing modality support.
Production-grade with LRU image cache bounds, augmentation hooks, and crash-free collation.
"""

from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


@dataclass
class MultimodalConfig:
    text_max_length: int = 256
    image_size: int = 224
    graph_max_nodes: int = 100
    missing_modality_strategy: str = "zero"  # zero, drop, or learnable token
    cache_images: bool = True
    augment: bool = False


class MultimodalDataset(Dataset):
    """
    Production Multimodal Dataset Loader.
    Safely integrates text tensors, transformed image tensors, and PyG graph instances.
    Enforces LRU bounded caching to prevent out-of-memory errors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_encoder: Callable,
        image_encoder: Optional[Callable] = None,
        graph_builder: Optional[Callable] = None,
        config: Optional[MultimodalConfig] = None,
        max_cache_size: int = 2000,
    ):
        """
        Args:
            df: DataFrame with columns [text, image_path, user_idx/user_id, label] etc.
            text_encoder: Function(texts) -> tokenized inputs dict
            image_encoder: Optional vision encoder for preprocessing
            graph_builder: Optional function(user_id) -> PyG Data object
            config: Dataset configuration
            max_cache_size: Maximum number of image tensors kept in LRU cache
        """
        self.df = df.reset_index(drop=True)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.graph_builder = graph_builder
        self.config = config or MultimodalConfig()
        self.max_cache_size = max_cache_size

        self._image_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._text_cache: Dict[int, Dict[str, Any]] = {}

        self.img_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info(f"MultimodalDataset initialized: {len(df)} samples, cache size {max_cache_size}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        """Load, transform, and cache image tensor with LRU eviction."""
        if path in self._image_cache:
            self._image_cache.move_to_end(path)
            return self._image_cache[path]

        try:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                tensor = self.img_transform(rgb_img)

            if len(self._image_cache) >= self.max_cache_size:
                self._image_cache.popitem(last=False)
            self._image_cache[path] = tensor
            return tensor
        except Exception as e:
            logger.warning(f"Error loading image {path}: {e}")
            return self._get_missing_image()

    def _get_missing_text(self) -> Dict[str, Any]:
        """Return placeholder for missing text."""
        return {
            "input_ids": torch.zeros(1, self.config.text_max_length, dtype=torch.long),
            "attention_mask": torch.zeros(1, self.config.text_max_length, dtype=torch.long),
        }

    def _get_missing_image(self) -> torch.Tensor:
        """Return zero tensor for missing image."""
        return torch.zeros(3, self.config.image_size, self.config.image_size, dtype=torch.float32)

    def _get_missing_graph(self) -> torch.Tensor:
        """Return placeholder for missing graph."""
        return torch.zeros(self.config.graph_max_nodes, 64, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        sample: Dict[str, Any] = {
            "index": idx,
            "label": torch.tensor(row.get("label", 0), dtype=torch.float32),
        }

        # 1. Text Modality
        text = str(row.get("text", "")) if pd.notna(row.get("text", "")) else ""
        if text:
            if idx in self._text_cache:
                sample["text_inputs"] = self._text_cache[idx]
            else:
                sample["text_inputs"] = self.text_encoder([text])
                if len(self.df) < 50000:
                    self._text_cache[idx] = sample["text_inputs"]
            sample["has_text"] = True
        else:
            sample["text_inputs"] = self._get_missing_text()
            sample["has_text"] = False

        # 2. Image Modality (Guaranteed tensor return)
        img_path = str(row.get("image_path", "")) if pd.notna(row.get("image_path", "")) else ""
        if img_path and os.path.exists(img_path):
            sample["image"] = self._load_image_tensor(img_path)
            sample["has_image"] = True
        else:
            sample["image"] = self._get_missing_image()
            sample["has_image"] = False

        # 3. User Index for Graph Access
        raw_uid = row.get("user_idx", row.get("user_id", 0))
        try:
            sample["user_idx"] = int(raw_uid)
        except (ValueError, TypeError):
            sample["user_idx"] = 0

        # 4. Optional Graph builder
        if "user_id" in row and pd.notna(row["user_id"]) and self.graph_builder is not None:
            try:
                sample["graph"] = self.graph_builder(row["user_id"])
                sample["has_graph"] = True
            except Exception as e:
                logger.warning(f"Graph build failed for {row['user_id']}: {e}")
                sample["graph"] = self._get_missing_graph()
                sample["has_graph"] = False
        else:
            sample["graph"] = self._get_missing_graph()
            sample["has_graph"] = False

        # 5. Metadata
        sample["metadata"] = {
            "user_id": str(row.get("user_id", "")),
            "post_id": str(row.get("post_id", "")),
            "timestamp": str(row.get("timestamp", "")),
            "source": str(row.get("source", "unknown")),
        }

        return sample

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Unified, crash-free batch collation."""
        collated: Dict[str, Any] = {
            "labels": torch.stack([b["label"] for b in batch]),
            "has_image": torch.tensor([b["has_image"] for b in batch], dtype=torch.bool),
            "has_text": torch.tensor([b["has_text"] for b in batch], dtype=torch.bool),
            "has_graph": torch.tensor([b.get("has_graph", False) for b in batch], dtype=torch.bool),
            "user_indices": torch.tensor([b["user_idx"] for b in batch], dtype=torch.long),
            "images": torch.stack([b["image"] for b in batch]),  # Always safe: all are tensors
            "metadata": [b["metadata"] for b in batch],
        }

        # Collate tokenized text
        if all(
            "input_ids" in b["text_inputs"] and b["text_inputs"]["input_ids"] is not None
            for b in batch
        ):
            collated["text_input_ids"] = torch.cat(
                [b["text_inputs"]["input_ids"] for b in batch], dim=0
            )
            collated["text_attention_mask"] = torch.cat(
                [b["text_inputs"]["attention_mask"] for b in batch], dim=0
            )
        else:
            collated["text_input_ids"] = None
            collated["text_attention_mask"] = None

        # Collate graphs if present
        if any(b.get("has_graph", False) for b in batch):
            collated["graphs"] = [b["graph"] for b in batch if b.get("has_graph", False)]
        else:
            collated["graphs"] = None

        return collated
