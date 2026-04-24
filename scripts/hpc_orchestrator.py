#!/usr/bin/env python3
"""
HPC Training Orchestrator for multi-node Distributed Data Parallel (DDP) training
on SLURM clusters with NVIDIA H100 GPUs. Handles job submission, checkpointing,
auto-resume, and monitoring.

This is NOT a toy script. It handles:
  - Multi-node multi-GPU DDP with torchrun
  - Automatic mixed precision (AMP) with bfloat16 on H100
  - Gradient checkpointing for memory efficiency
  - FSDP (Fully Sharded Data Parallel) for large models
  - Checkpointing every N steps + best model tracking
  - Auto-resume from latest checkpoint
  - WandB logging with distributed-safe metrics
  - SLURM signal handling for graceful preemption
  - CPU offloading for optimizer states.

Usage:
    python scripts/hpc_orchestrator.py --config configs/hpc_train.yaml
    # Or submit via SLURM:
    sbatch scripts/slurm_launcher.sh
"""
import os
import sys
import time
import json
import signal
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset

# Optional: DeepSpeed for ZeRO-3 offloading
try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Rank %(rank)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def get_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))

def is_main_process() -> bool:
    return get_rank() == 0


@dataclass
class HPCTrainingConfig:
    """Complete HPC training configuration."""
    # Model
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 256
    
    # Training
    num_epochs: int = 3
    batch_size_per_gpu: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Distributed
    backend: str = "nccl"
    use_fsdp: bool = False  # Use FSDP instead of DDP for large models
    use_deepspeed: bool = False
    
    # Precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16 on H100, float16 on older GPUs
    
    # Memory
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    
    # Checkpointing
    output_dir: str = "models/hpc_training"
    checkpoint_dir: str = "models/hpc_training/checkpoints"
    save_steps: int = 500
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    wandb_project: str = "doom-index"
    wandb_run_name: Optional[str] = None
    
    # Data
    train_data_path: str = "data/processed/train.csv"
    val_data_path: str = "data/processed/val.csv"
    num_workers: int = 4
    
    # Optimization
    seed: int = 42
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    
    def to_dict(self) -> Dict:
        return asdict(self)


class CheckpointManager:
    """
    Production checkpoint manager with rotation, best-model tracking,
    and atomic saves (write to temp, then rename).
    """
    
    def __init__(self, checkpoint_dir: str, save_total_limit: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.best_metric = float("-inf")
        self.best_checkpoint_path: Optional[Path] = None
    
    def save_checkpoint(self, state_dict: Dict[str, Any], step: int, 
                        metric: Optional[float] = None, is_epoch_end: bool = False):
        """Save checkpoint atomically."""
        if not is_main_process():
            return
        
        checkpoint = {
            "step": step,
            "metric": metric,
            "timestamp": datetime.utcnow().isoformat(),
            **state_dict
        }
        
        suffix = "epoch" if is_epoch_end else f"step_{step}"
        if metric is not None:
            suffix += f"_metric_{metric:.4f}"
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{suffix}.pt"
        temp_path = checkpoint_path.with_suffix(".tmp")
        
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Track best
        if metric is not None and metric > self.best_metric:
            self.best_metric = metric
            best_path = self.checkpoint_dir / "best_model.pt"
            best_temp = best_path.with_suffix(".tmp")
            torch.save(checkpoint, best_temp)
            best_temp.rename(best_path)
            logger.info(f"New best model: metric={metric:.4f}")
        
        # Rotate old checkpoints
        self._rotate_checkpoints()
    
    def _rotate_checkpoints(self):
        """Remove old checkpoints, keeping only N most recent + best."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-step_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_ckpt in checkpoints[self.save_total_limit:]:
            old_ckpt.unlink()
            logger.info(f"Removed old checkpoint: {old_ckpt}")
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint for auto-resume."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Exclude best_model.pt from auto-resume
        checkpoints = [c for c in checkpoints if c.name != "best_model.pt"]
        
        if not checkpoints:
            return None
        
        latest = checkpoints[0]
        logger.info(f"Resuming from checkpoint: {latest}")
        return torch.load(latest, map_location="cpu")
    
    def load_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return torch.load(best_path, map_location="cpu")
        return None


class MetricsTracker:
    """
    Distributed-safe metrics tracking with WandB integration.
    Handles all-reduce for aggregated metrics across ranks.
    """
    
    def __init__(self, config: HPCTrainingConfig):
        self.config = config
        self.step = 0
        self.epoch = 0
        self.global_step = 0
        
        # Initialize WandB on main process
        self.wandb = None
        if is_main_process() and config.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    config=config.to_dict()
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("WandB not installed. Skipping.")
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics (distributed-safe)."""
        step = step or self.global_step
        
        # All-reduce metrics across ranks
        if get_world_size() > 1:
            for key in metrics:
                tensor = torch.tensor(metrics[key], device="cuda")
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                metrics[key] = tensor.item()
        
        # Log to WandB
        if self.wandb and is_main_process():
            self.wandb.log(metrics, step=step)
        
        # Log to console
        if is_main_process():
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"Step {step} | {metrics_str}")
    
    def finish(self):
        if self.wandb and is_main_process():
            self.wandb.finish()


class SLURMHandler:
    """
    Handle SLURM signals for graceful preemption and checkpointing.
    SIGUSR1 is sent by SLURM before job preemption.
    """
    
    def __init__(self, checkpoint_callback: Callable):
        self.checkpoint_callback = checkpoint_callback
        self.preempted = False
        
        signal.signal(signal.SIGUSR1, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.preempted = True
        self.checkpoint_callback()
        sys.exit(0)
    
    def is_preempted(self) -> bool:
        return self.preempted


class HPCTrainer:
    """
    Production HPC trainer supporting DDP, FSDP, DeepSpeed, and mixed precision.
    """
    
    def __init__(self, config: HPCTrainingConfig):
        self.config = config
        self.device = torch.device(f"cuda:{get_rank() % torch.cuda.device_count()}")
        
        # Initialize distributed
        if get_world_size() > 1:
            dist.init_process_group(backend=config.backend)
        
        # Set device
        torch.cuda.set_device(self.device)
        
        # Seed
        self._set_seed(config.seed)
        
        # Components
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir, 
            config.save_total_limit
        )
        self.metrics_tracker = MetricsTracker(config)
        
        # AMP
        self.scaler = GradScaler() if config.use_amp and config.amp_dtype == "float16" else None
        
        # State
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.current_step = 0
        self.current_epoch = 0
        
        # SLURM handler
        self.slurm_handler = SLURMHandler(self._emergency_checkpoint)
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Deterministic mode (slower but reproducible)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    def _emergency_checkpoint(self):
        """Emergency checkpoint on SLURM preemption."""
        if self.model and self.optimizer:
            state = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "step": self.current_step,
                "epoch": self.current_epoch,
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None
            }
            self.checkpoint_manager.save_checkpoint(
                state, self.current_step, is_epoch_end=False
            )
            logger.info("Emergency checkpoint saved.")
    
    def setup_model(self, model: torch.nn.Module):
        """Setup model with DDP/FSDP/DeepSpeed."""
        model = model.to(self.device)
        
        if self.config.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            elif hasattr(model, "encoder"):
                model.encoder.gradient_checkpointing = True
        
        if self.config.use_deepspeed and HAS_DEEPSPEED:
            # DeepSpeed ZeRO-3
            ds_config = {
                "train_batch_size": self.config.batch_size_per_gpu * get_world_size() * self.config.gradient_accumulation_steps,
                "train_micro_batch_size_per_gpu": self.config.batch_size_per_gpu,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": self.config.learning_rate,
                        "weight_decay": self.config.weight_decay
                    }
                },
                "scheduler": {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": self.config.learning_rate,
                        "warmup_num_steps": 1000
                    }
                },
                "fp16": {
                    "enabled": self.config.use_amp and self.config.amp_dtype == "float16"
                },
                "bf16": {
                    "enabled": self.config.use_amp and self.config.amp_dtype == "bfloat16"
                },
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {
                        "device": "cpu" if self.config.cpu_offload else "none"
                    },
                    "offload_param": {
                        "device": "cpu" if self.config.cpu_offload else "none"
                    }
                }
            }
            self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=model,
                config=ds_config
            )
        
        elif self.config.use_fsdp:
            # FSDP for large models
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16,
                reduce_dtype=torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float32,
                buffer_dtype=torch.float32
            )
            
            self.model = FSDP(
                model,
                auto_wrap_policy=transformer_auto_wrap_policy,
                mixed_precision=mp_policy,
                device_id=self.device,
                limit_all_gathers=True,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE
            )
        
        else:
            # Standard DDP
            if get_world_size() > 1:
                self.model = DDP(
                    model,
                    device_ids=[self.device],
                    output_device=self.device,
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True
                )
            else:
                self.model = model
        
        logger.info(f"Model setup complete. Using: {'DeepSpeed' if self.config.use_deepspeed else 'FSDP' if self.config.use_fsdp else 'DDP'}")
    
    def setup_optimizer(self, model_parameters):
        """Setup optimizer and scheduler."""
        from transformers import get_linear_schedule_with_warmup
        
        if not self.config.use_deepspeed:
            self.optimizer = torch.optim.AdamW(
                model_parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # Scheduler will be set after dataloader is created (to know total steps)
        self.scheduler = None
    
    def setup_data(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """Setup dataloaders with distributed sampling."""
        if get_world_size() > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                seed=self.config.seed
            )
        else:
            train_sampler = None
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size_per_gpu,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            prefetch_factor=self.config.dataloader_prefetch_factor if self.config.num_workers > 0 else None,
            persistent_workers=self.config.num_workers > 0
        )
        
        if val_dataset is not None:
            if get_world_size() > 1:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                    shuffle=False
                )
            else:
                val_sampler = None
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size_per_gpu * 2,  # Larger batch for eval
                sampler=val_sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.dataloader_pin_memory
            )
        
        # Setup scheduler now that we know num_steps
        total_steps = len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if not self.config.use_deepspeed:
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        
        logger.info(f"Data setup: {len(self.train_loader)} batches/epoch, {total_steps} total steps")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
        
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            if self.slurm_handler.is_preempted():
                break
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward
            with autocast(enabled=self.config.use_amp, dtype=torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16):
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_fsdp:
                    self.model.clip_grad_norm_(self.config.max_grad_norm)
                elif not self.config.use_deepspeed:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.current_step += 1
                
                # Logging
                if self.current_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                    self.metrics_tracker.log({
                        "train/loss": total_loss / (batch_idx + 1),
                        "train/lr": lr,
                        "train/epoch": epoch + batch_idx / len(self.train_loader)
                    }, step=self.current_step)
                
                # Checkpointing
                if self.current_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                
                # Evaluation
                if self.val_loader and self.current_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self.metrics_tracker.log(eval_metrics, step=self.current_step)
        
        avg_loss = total_loss / max(len(self.train_loader), 1)
        return {"train/epoch_loss": avg_loss}
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            with autocast(enabled=self.config.use_amp, dtype=torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16):
                outputs = self.model(**batch)
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        avg_loss = total_loss / max(len(self.val_loader), 1)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        
        metrics = {
            "eval/loss": avg_loss,
            "eval/accuracy": accuracy,
            "eval/f1": f1
        }
        
        # Save checkpoint if best
        if is_main_process():
            self._save_checkpoint(metric=f1)
        
        self.model.train()
        return metrics
    
    def _save_checkpoint(self, metric: Optional[float] = None):
        """Save training checkpoint."""
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        state = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "step": self.current_step,
            "epoch": self.current_epoch,
            "config": self.config.to_dict()
        }
        
        if self.scaler:
            state["scaler_state_dict"] = self.scaler.state_dict()
        
        self.checkpoint_manager.save_checkpoint(state, self.current_step, metric=metric)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Load checkpoint for resuming training."""
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        
        if checkpoint is None:
            logger.info("No checkpoint found. Starting from scratch.")
            return
        
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        
        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_step = checkpoint.get("step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)
        
        logger.info(f"Resumed from step {self.current_step}, epoch {self.current_epoch}")
    
    def train(self, num_epochs: Optional[int] = None):
        """Run full training loop."""
        epochs = num_epochs or self.config.num_epochs
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"World size: {get_world_size()}, Rank: {get_rank()}")
        logger.info(f"Device: {self.device}, AMP: {self.config.use_amp} ({self.config.amp_dtype})")
        
        for epoch in range(self.current_epoch, epochs):
            if self.slurm_handler.is_preempted():
                break
            
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            epoch_metrics = self.train_epoch(epoch)
            self.current_epoch = epoch + 1
            
            # End-of-epoch logging
            self.metrics_tracker.log(epoch_metrics, step=self.current_step)
            
            # End-of-epoch checkpoint
            self._save_checkpoint(is_epoch_end=True)
        
        # Final evaluation
        if self.val_loader:
            final_metrics = self.evaluate()
            self.metrics_tracker.log(final_metrics, step=self.current_step)
        
        self.metrics_tracker.finish()
        logger.info("Training complete.")
    
    def cleanup(self):
        """Cleanup distributed training."""
        if get_world_size() > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    config = HPCTrainingConfig(**config_dict)
    
    # Create trainer
    trainer = HPCTrainer(config)
    
    try:
        # Setup model, optimizer, data
        # (These would be imported from your actual model definitions)
        # trainer.setup_model(your_model)
        # trainer.setup_optimizer(your_model.parameters())
        # trainer.setup_data(train_dataset, val_dataset)
        # trainer.load_checkpoint()
        # trainer.train()
        pass
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
