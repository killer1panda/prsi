#!/usr/bin/env python3
"""Unified Experiment Tracking for H100 Cluster Training.

Integrates MLflow and Weights & Biases with automatic fallback,
providing a single interface for logging metrics, parameters,
artifacts, and model versions across distributed training jobs.

Features:
- Automatic backend detection (MLflow URI > WandB API key > local filesystem)
- Distributed-safe logging with rank-0 gating
- GPU utilization tracking (memory, temperature, power draw)
- Custom metric aggregation across workers
- Artifact versioning (checkpoints, datasets, config files)
- Hyperparameter search integration (Optuna, Ray Tune)
- Model registry staging (None -> Staging -> Production)

Usage:
    tracker = ExperimentTracker(experiment_name="doom_distilbert_h100")
    tracker.log_params({"lr": 2e-5, "batch_size": 32})
    tracker.log_metrics({"train_loss": 0.42}, step=epoch)
    tracker.save_model(model, "checkpoint_epoch_5", metadata={"f1": 0.89})
"""

import json
import logging
import os
import socket
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Optional MLflow
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.debug("MLflow not installed. Install: pip install mlflow>=2.8")

# Optional WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.debug("WandB not installed. Install: pip install wandb>=0.16")

# GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class GPUStats:
    """Snapshot of GPU telemetry."""
    index: int
    utilization_gpu: float  # Percentage
    utilization_mem: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float
    power_draw_w: float
    power_limit_w: float
    clock_sm_mhz: float
    clock_mem_mhz: float
    throttled: bool


class ExperimentTracker:
    """Production experiment tracker with H100-specific telemetry."""

    def __init__(
        self,
        experiment_name: str = "doom_index",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        local_dir: str = "./mlruns",
        auto_init: bool = True,
        log_gpu_stats: bool = True,
        log_system_metrics: bool = True,
        rank: Optional[int] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name or f"{experiment_name}_{socket.gethostname()}_{int(time.time())}"
        self.local_dir = Path(local_dir)
        self.log_gpu_stats = log_gpu_stats and PYNVML_AVAILABLE
        self.log_system_metrics = log_system_metrics
        self._rank = rank
        self._world_size = 1
        self._active = False
        self._gpu_stats_history: List[Dict] = []

        # Detect distributed rank
        if self._rank is None:
            if dist.is_available() and dist.is_initialized():
                self._rank = dist.get_rank()
                self._world_size = dist.get_world_size()
            else:
                self._rank = 0

        self._is_master = self._rank == 0

        # Initialize backends
        self._mlflow_active = False
        self._wandb_active = False
        self._local_active = False

        if auto_init and self._is_master:
            self._init_mlflow(tracking_uri)
            self._init_wandb(wandb_project, wandb_entity)
            if not self._mlflow_active and not self._wandb_active:
                self._init_local()

        self._active = True
        self._step = 0
        self._epoch = 0

    def _init_mlflow(self, tracking_uri: Optional[str]) -> None:
        """Initialize MLflow tracking server or local backend."""
        if not MLFLOW_AVAILABLE:
            return
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", f"file://{self.local_dir.absolute()}")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        self._mlflow_active = True
        logger.info(f"MLflow tracking: {uri} | Run: {mlflow.active_run().info.run_id}")

    def _init_wandb(self, project: Optional[str], entity: Optional[str]) -> None:
        """Initialize Weights & Biases run."""
        if not WANDB_AVAILABLE:
            return
        if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline":
            wandb.init(
                project=project or self.experiment_name,
                entity=entity,
                name=self.run_name,
                config={},
            )
            self._wandb_active = True
            logger.info(f"WandB tracking: {wandb.run.url if wandb.run else 'offline'}")

    def _init_local(self) -> None:
        """Fallback to JSONL local logging."""
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self._local_log_path = self.local_dir / f"{self.run_name}.jsonl"
        self._local_active = True
        logger.info(f"Local tracking: {self._local_log_path}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters and configuration."""
        if not self._is_master:
            return
        clean_params = {k: self._serialize(v) for k, v in params.items()}
        if self._mlflow_active:
            mlflow.log_params(clean_params)
        if self._wandb_active:
            wandb.config.update(clean_params)
        if self._local_active:
            self._local_write({"type": "params", "data": clean_params})
        logger.debug(f"Logged params: {list(clean_params.keys())}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = "") -> None:
        """Log scalar metrics with optional step indexing."""
        if not self._is_master:
            return
        step = step if step is not None else self._step
        prefixed = {f"{prefix}{k}" if prefix else k: float(v) for k, v in metrics.items()}

        if self._mlflow_active:
            mlflow.log_metrics(prefixed, step=step)
        if self._wandb_active:
            wandb.log(prefixed, step=step)
        if self._local_active:
            self._local_write({"type": "metrics", "step": step, "data": prefixed})

        # GPU telemetry on every 10th call to avoid overhead
        if self.log_gpu_stats and step % 10 == 0:
            gpu_stats = self._read_gpu_stats()
            if gpu_stats:
                flat = {}
                for g in gpu_stats:
                    for k, v in asdict(g).items():
                        if k != "index":
                            flat[f"gpu_{g.index}_{k}"] = v
                self.log_metrics(flat, step=step, prefix="hardware/")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log file artifact (checkpoint, config, dataset sample)."""
        if not self._is_master:
            return
        path = Path(local_path)
        if not path.exists():
            logger.warning(f"Artifact not found: {local_path}")
            return
        if self._mlflow_active:
            mlflow.log_artifact(str(path), artifact_path)
        if self._wandb_active:
            wandb.save(str(path), base_path=str(path.parent))
        if self._local_active:
            dest = self.local_dir / "artifacts" / (artifact_path or "") / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(str(path), str(dest))

    def save_model(
        self,
        model: torch.nn.Module,
        name: str,
        metadata: Optional[Dict] = None,
        register: bool = False,
    ) -> Optional[str]:
        """Save model checkpoint and optionally register to model registry.

        Returns:
            Model version string if registered, else None.
        """
        if not self._is_master:
            return None

        # Save to temporary path
        save_dir = Path(tempfile.gettempdir()) / f"doom_model_{name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "metadata": metadata or {},
            "tracker_run": self.run_name,
        }, save_dir / "pytorch_model.bin")

        version = None
        if self._mlflow_active:
            mlflow.pytorch.log_model(model, artifact_path=f"models/{name}")
            if register and MLFLOW_AVAILABLE:
                client = MlflowClient()
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/models/{name}"
                mv = client.create_model_version(
                    name=self.experiment_name,
                    source=model_uri,
                    run_id=run_id,
                    tags=metadata or {},
                )
                version = mv.version
                logger.info(f"Model registered: {self.experiment_name} v{version}")
        if self._wandb_active:
            wandb.save(str(save_dir / "pytorch_model.bin"))

        self.log_artifact(str(save_dir / "pytorch_model.bin"), f"models/{name}")
        return version

    def log_dataset_profile(self, dataset_path: str, split: str = "train") -> None:
        """Log dataset statistics as a table artifact."""
        import pandas as pd
        df = pd.read_parquet(dataset_path) if dataset_path.endswith(".parquet") else pd.read_csv(dataset_path)
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "label_balance": df["label"].value_counts().to_dict() if "label" in df.columns else {},
            "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }
        self.log_metrics({f"data/{split}_{k}": v for k, v in stats.items()})

    def _read_gpu_stats(self) -> List[GPUStats]:
        """Read per-GPU telemetry via NVML."""
        if not PYNVML_AVAILABLE:
            return []
        stats = []
        for i in range(torch.cuda.device_count()):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                clocks = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle) != 0

                stats.append(GPUStats(
                    index=i,
                    utilization_gpu=util.gpu,
                    utilization_mem=util.memory,
                    memory_used_mb=mem.used / (1024 * 1024),
                    memory_total_mb=mem.total / (1024 * 1024),
                    temperature_c=temp,
                    power_draw_w=power,
                    power_limit_w=power_limit,
                    clock_sm_mhz=clocks,
                    clock_mem_mhz=mem_clock,
                    throttled=throttle,
                ))
            except Exception as e:
                logger.debug(f"GPU stat error on device {i}: {e}")
        return stats

    def _serialize(self, value: Any) -> Union[str, float, int]:
        """Convert non-serializable types for logging backends."""
        if isinstance(value, (int, float, str, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return json.dumps(value)
        if isinstance(value, dict):
            return json.dumps(value)
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist())
        if isinstance(value, torch.Tensor):
            return json.dumps(value.detach().cpu().numpy().tolist())
        return str(value)

    def _local_write(self, record: Dict) -> None:
        if self._local_active:
            with open(self._local_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def finish(self) -> None:
        """Finalize the experiment run."""
        if not self._is_master:
            return
        if self._mlflow_active:
            mlflow.end_run()
        if self._wandb_active:
            wandb.finish()
        self._active = False
        logger.info("Experiment tracking finished")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


# Convenience factory for H100 cluster jobs
def create_h100_tracker(
    job_name: str,
    nodes: int = 1,
    gpus_per_node: int = 4,
    mlflow_uri: Optional[str] = None,
) -> ExperimentTracker:
    """Create tracker pre-configured for H100 cluster job."""
    tracker = ExperimentTracker(
        experiment_name=f"doom_h100_{job_name}",
        run_name=f"{job_name}_n{nodes}x{gpus_per_node}_{int(time.time())}",
        tracking_uri=mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI"),
        log_gpu_stats=True,
        log_system_metrics=True,
    )
    tracker.log_params({
        "hardware": "H100",
        "num_nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": nodes * gpus_per_node,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    })
    return tracker


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with create_h100_tracker("test_run", nodes=1, gpus_per_node=1) as tracker:
        tracker.log_params({"lr": 2e-5, "model": "distilbert"})
        for step in range(10):
            tracker.log_metrics({"loss": 1.0 / (step + 1), "acc": step * 0.1}, step=step)
            time.sleep(0.1)
        print("Tracking test complete")
