"""Common utilities for Doom Index.

Device detection, seed setting, checkpointing, and logging setup.
"""

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """Setup structured logging."""
    handlers = [logging.StreamHandler()]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger = logging.getLogger(__name__)
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        return device
    return torch.device("cpu")


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: dict,
    path: str,
    scheduler=None,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cuda"):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


def count_parameters(model) -> tuple:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_gpu_memory():
    """Get GPU memory usage."""
    if torch.cuda.is_available():
        return {
            i: {
                "allocated": torch.cuda.memory_allocated(i) / 1e9,
                "reserved": torch.cuda.memory_reserved(i) / 1e9,
                "total": torch.cuda.get_device_properties(i).total_memory / 1e9,
            }
            for i in range(torch.cuda.device_count())
        }
    return {}


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
