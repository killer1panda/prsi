"""
Multimodal dataset loader handling text, image, and graph data with missing modality support.
Production-grade with caching, augmentation hooks, and efficient collation.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

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
    Dataset for multimodal Doom Index training.
    Handles text (tokenized), image (PIL/Path), and graph (PyG Data) modalities.
    """

    def __init__(self, 
                 df: pd.DataFrame,
                 text_encoder: Callable,
                 image_encoder: Optional[Callable] = None,
                 graph_builder: Optional[Callable] = None,
                 config: Optional[MultimodalConfig] = None):
        """
        Args:
            df: DataFrame with columns [text, image_path, user_id, label] etc.
            text_encoder: Function(texts) -> tokenized inputs dict
            image_encoder: Optional vision encoder for preprocessing
            graph_builder: Optional function(user_id) -> PyG Data object
            config: Dataset configuration
        """
        self.df = df.reset_index(drop=True)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.graph_builder = graph_builder
        self.config = config or MultimodalConfig()

        self._image_cache: Dict[str, Image.Image] = {}
        self._text_cache: Dict[int, Dict] = {}

        logger.info(f"MultimodalDataset initialized: {len(df)} samples")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        row = self.df.iloc[idx]
        sample = {"index": idx, "label": torch.tensor(row.get("label", 0), dtype=torch.float32)}

        # Text modality
        if "text" in row and pd.notna(row["text"]):
            if idx in self._text_cache:
                sample["text_inputs"] = self._text_cache[idx]
            else:
                text = str(row["text"])
                sample["text_inputs"] = self.text_encoder([text])
                if len(self.df) < 100000:  # Cache only for smaller datasets
                    self._text_cache[idx] = sample["text_inputs"]
        else:
            sample["text_inputs"] = self._get_missing_text()

        # Image modality
        if "image_path" in row and pd.notna(row["image_path"]):
            img_path = str(row["image_path"])
            sample["image"] = self._load_image(img_path)
            sample["has_image"] = True
        else:
            sample["image"] = self._get_missing_image()
            sample["has_image"] = False

        # Graph modality
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

        # Metadata
        sample["metadata"] = {
            "user_id": row.get("user_id", ""),
            "post_id": row.get("post_id", ""),
            "timestamp": row.get("timestamp", ""),
            "source": row.get("source", "unknown")
        }

        return sample

    def _load_image(self, path: str) -> Union[torch.Tensor, Image.Image]:
        """Load and optionally preprocess image."""
        if path in self._image_cache:
            return self._image_cache[path]

        try:
            img = Image.open(path).convert("RGB")
            if self.config.cache_images:
                self._image_cache[path] = img
            return img
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            return self._get_missing_image()

    def _get_missing_text(self) -> Dict:
        """Return placeholder for missing text."""
        if self.config.missing_modality_strategy == "zero":
            return {
                "input_ids": torch.zeros(1, self.config.text_max_length, dtype=torch.long),
                "attention_mask": torch.zeros(1, self.config.text_max_length, dtype=torch.long)
            }
        else:
            # Return a learnable token representation handled by model
            return {"input_ids": None, "attention_mask": None, "missing": True}

    def _get_missing_image(self) -> torch.Tensor:
        """Return zero tensor for missing image."""
        return torch.zeros(3, self.config.image_size, self.config.image_size)

    def _get_missing_graph(self) -> torch.Tensor:
        """Return placeholder for missing graph."""
        return torch.zeros(self.config.graph_max_nodes, 64)  # Assumed node feature dim

    def collate_fn(self, batch: List[Dict]) -> Dict[str, any]:
        """
        Custom collation for multimodal batches.
        Handles variable-length sequences and missing modalities.
        """
        collated = {
            "labels": torch.stack([b["label"] for b in batch]),
            "has_image": torch.tensor([b["has_image"] for b in batch], dtype=torch.bool),
            "has_graph": torch.tensor([b["has_graph"] for b in batch], dtype=torch.bool),
            "metadata": [b["metadata"] for b in batch]
        }

        # Collate text
        if all("input_ids" in b["text_inputs"] and b["text_inputs"]["input_ids"] is not None for b in batch):
            collated["text_input_ids"] = torch.cat([b["text_inputs"]["input_ids"] for b in batch], dim=0)
            collated["text_attention_mask"] = torch.cat([b["text_inputs"]["attention_mask"] for b in batch], dim=0)
        else:
            collated["text_input_ids"] = None
            collated["text_missing"] = True

        # Collate images
        if any(b["has_image"] for b in batch):
            images = []
            for b in batch:
                if b["has_image"] and self.image_encoder is not None:
                    # Preprocess on-the-fly if encoder provided
                    img_tensor = self.image_encoder.preprocess([b["image"]])
                    images.append(img_tensor[0])
                else:
                    images.append(b["image"])
            collated["images"] = torch.stack(images)
        else:
            collated["images"] = None

        # Collate graphs (batching handled by PyG if Data objects, else pad)
        if any(b["has_graph"] for b in batch):
            graphs = [b["graph"] for b in batch if b["has_graph"]]
            collated["graphs"] = graphs
        else:
            collated["graphs"] = None

        return collated
