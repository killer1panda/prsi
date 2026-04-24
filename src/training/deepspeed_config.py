#!/usr/bin/env python3
"""DeepSpeed + FSDP Distributed Training Configuration for H100 Clusters.

Provides:
- ZeRO-3 configuration with parameter/optimizer/state partitioning
- CPU offloading for large models exceeding H100 80GB VRAM
- Gradient checkpointing activation recomputation
- Mixed precision (bf16) optimal for H100 Tensor Cores
- NVLink-aware communication buckets
- Pipeline parallelism for giant models.

Usage:
    deepspeed --num_gpus=4 src/training/distributed_trainer.py \
        --deepspeed configs/training/ds_config_h100.json
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class DeepSpeedConfig:
    """Immutable DeepSpeed configuration object for H100 training."""

    # Training
    train_batch_size: int = 256  # Global batch across all GPUs
    train_micro_batch_size_per_gpu: int = 32  # Per-GPU batch (H100 can handle 32-64)
    gradient_accumulation_steps: int = 2
    gradient_clipping: float = 1.0

    # Optimizer
    optimizer_type: str = "AdamW"
    lr: float = 2e-5
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.01

    # Scheduler
    lr_scheduler_type: str = "WarmupDecayLR"
    warmup_min_lr: float = 0.0
    warmup_max_lr: float = 2e-5
    warmup_num_steps: int = 500
    total_num_steps: int = 10000

    # Precision (H100: bf16 is faster and more stable than fp16)
    bf16: Dict[str, bool] = field(default_factory=lambda: {"enabled": True})
    fp16: Dict[str, bool] = field(default_factory=lambda: {"enabled": False})
    amp: Dict[str, bool] = field(default_factory=lambda: {"enabled": False})

    # ZeRO Optimization (H100: ZeRO-3 with aggressive partitioning)
    zero_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
            "ratio": 1.0,
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
            "ratio": 1.0,
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    })

    # Communication (tuned for NVLink H100 nodes)
    communication_options: Dict[str, Any] = field(default_factory=lambda: {
        "bucket_size": 5e8,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": True,
        "overlap_comm": True,
    })

    # Activation checkpointing (trade compute for memory)
    activation_checkpointing: Dict[str, Any] = field(default_factory=lambda: {
        "partition_activations": True,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": True,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False,
    })

    # Flops profiler (for H100 utilization analysis)
    flops_profiler: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "profile_step": 50,
        "detailed": True,
        "output_file": "logs/flops_profiler.log",
    })

    # Logging
    steps_per_print: int = 100
    wall_clock_breakdown: bool = True
    dump_state: bool = False

    # Checkpointing
    checkpoint: Dict[str, Any] = field(default_factory=lambda: {
        "tag": "doom_checkpoint",
        "load_universal": False,
    })

    def to_json(self, path: str) -> None:
        """Serialize configuration to DeepSpeed-compatible JSON."""
        config_dict = asdict(self)
        # DeepSpeed expects specific nested structure
        output = {
            "train_batch_size": config_dict["train_batch_size"],
            "train_micro_batch_size_per_gpu": config_dict["train_micro_batch_size_per_gpu"],
            "gradient_accumulation_steps": config_dict["gradient_accumulation_steps"],
            "gradient_clipping": config_dict["gradient_clipping"],
            "optimizer": {
                "type": config_dict["optimizer_type"],
                "params": {
                    "lr": config_dict["lr"],
                    "betas": config_dict["betas"],
                    "eps": config_dict["eps"],
                    "weight_decay": config_dict["weight_decay"],
                },
            },
            "scheduler": {
                "type": config_dict["lr_scheduler_type"],
                "params": {
                    "warmup_min_lr": config_dict["warmup_min_lr"],
                    "warmup_max_lr": config_dict["warmup_max_lr"],
                    "warmup_num_steps": config_dict["warmup_num_steps"],
                    "total_num_steps": config_dict["total_num_steps"],
                },
            },
            "bf16": config_dict["bf16"],
            "fp16": config_dict["fp16"],
            "zero_optimization": config_dict["zero_optimization"],
            "activation_checkpointing": config_dict["activation_checkpointing"],
            "flops_profiler": config_dict["flops_profiler"],
            "steps_per_print": config_dict["steps_per_print"],
            "wall_clock_breakdown": config_dict["wall_clock_breakdown"],
            "checkpoint": config_dict["checkpoint"],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"DeepSpeed config written to {path}")

    @classmethod
    def for_h100_4gpu(cls, model_size: str = "distilbert") -> "DeepSpeedConfig":
        """Factory method for 4x H100 80GB node configuration.

        Args:
            model_size: One of "distilbert", "bert_base", "bert_large"
        """
        configs = {
            "distilbert": {
                "train_micro_batch_size_per_gpu": 64,
                "gradient_accumulation_steps": 2,
                "zero_optimization": {
                    "stage": 2,  # DistilBERT fits in H100 VRAM, ZeRO-2 sufficient
                    "offload_optimizer": {"device": "none"},
                    "offload_param": {"device": "none"},
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": 1e9,
                },
            },
            "bert_base": {
                "train_micro_batch_size_per_gpu": 32,
                "gradient_accumulation_steps": 4,
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    "offload_param": {"device": "none"},
                    "overlap_comm": True,
                    "reduce_bucket_size": 5e8,
                },
            },
            "bert_large": {
                "train_micro_batch_size_per_gpu": 16,
                "gradient_accumulation_steps": 8,
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    "offload_param": {"device": "cpu", "pin_memory": True},
                    "overlap_comm": True,
                    "reduce_bucket_size": 2e8,
                    "stage3_prefetch_bucket_size": 2e8,
                },
            },
        }
        base = cls()
        if model_size in configs:
            overrides = configs[model_size]
            for key, val in overrides.items():
                setattr(base, key, val)
        base.train_batch_size = (
            base.train_micro_batch_size_per_gpu
            * base.gradient_accumulation_steps
            * 4  # 4 GPUs
        )
        logger.info(f"H100 4-GPU config generated for {model_size}: "
                    f"global_batch={base.train_batch_size}")
        return base

    @classmethod
    def for_h100_multinode(cls, nodes: int = 2, gpus_per_node: int = 4, model_size: str = "distilbert") -> "DeepSpeedConfig":
        """Factory method for multi-node H100 cluster.

        Args:
            nodes: Number of compute nodes
            gpus_per_node: GPUs per node (typically 4 or 8 for H100 DGX)
            model_size: Model architecture identifier
        """
        total_gpus = nodes * gpus_per_node
        base = cls.for_h100_4gpu(model_size)
        # Scale global batch but keep per-GPU micro batch constant
        base.train_batch_size = (
            base.train_micro_batch_size_per_gpu
            * base.gradient_accumulation_steps
            * total_gpus
        )
        # Tune communication for inter-node IB/Ethernet
        base.zero_optimization["reduce_bucket_size"] = 2e8
        base.zero_optimization["stage3_prefetch_bucket_size"] = 2e8
        base.communication_options["allgather_bucket_size"] = 2e8
        logger.info(f"Multi-node H100 config: {nodes}x{gpus_per_node} GPUs, "
                    f"global_batch={base.train_batch_size}")
        return base


class FSDPConfig:
    """PyTorch FSDP configuration wrapper for H100 clusters.

    FSDP is preferred over DeepSpeed when:
    - You need finer-grained control over wrapping policies
    - Model fits mostly in H100 VRAM with activation checkpointing
    - Using PyTorch-native features (compile, flex_attention)
    """

    def __init__(
        self,
        sharding_strategy: str = "FULL_SHARD",  # or SHARD_GRAD_OP, NO_SHARD
        backward_prefetch: str = "BACKWARD_PRE",  # or BACKWARD_POST
        cpu_offload: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = True,
        sync_module_states: bool = False,
        activation_checkpointing: bool = True,
        mixed_precision: str = "bf16",  # H100 optimal
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.limit_all_gathers = limit_all_gathers
        self.use_orig_params = use_orig_params
        self.sync_module_states = sync_module_states
        self.activation_checkpointing = activation_checkpointing
        self.mixed_precision = mixed_precision

    def get_policy(self):
        """Return PyTorch FSDP constructor kwargs."""
        from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
        return {
            "sharding_strategy": ShardingStrategy[self.sharding_strategy],
            "backward_prefetch": BackwardPrefetch[self.backward_prefetch],
            "cpu_offload": self.cpu_offload,
            "limit_all_gathers": self.limit_all_gathers,
            "use_orig_params": self.use_orig_params,
            "sync_module_states": self.sync_module_states,
            "mixed_precision": self._get_mp_policy(),
        }

    def _get_mp_policy(self):
        from torch.distributed.fsdp import MixedPrecision
        if self.mixed_precision == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self.mixed_precision == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        return None


def generate_all_configs(output_dir: str = "configs/training") -> List[str]:
    """Generate the complete suite of H100 training configurations.

    Returns:
        List of generated file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generated = []

    scenarios = [
        ("ds_h100_4gpu_distilbert.json", DeepSpeedConfig.for_h100_4gpu("distilbert")),
        ("ds_h100_4gpu_bert_base.json", DeepSpeedConfig.for_h100_4gpu("bert_base")),
        ("ds_h100_4gpu_bert_large.json", DeepSpeedConfig.for_h100_4gpu("bert_large")),
        ("ds_h100_2node_distilbert.json", DeepSpeedConfig.for_h100_multinode(2, 4, "distilbert")),
        ("ds_h100_4node_distilbert.json", DeepSpeedConfig.for_h100_multinode(4, 4, "distilbert")),
    ]

    for filename, config in scenarios:
        path = os.path.join(output_dir, filename)
        config.to_json(path)
        generated.append(path)

    logger.info(f"Generated {len(generated)} DeepSpeed configurations in {output_dir}")
    return generated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = generate_all_configs()
    for p in paths:
        print(f"  {p}")
