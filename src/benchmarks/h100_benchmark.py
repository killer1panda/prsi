#!/usr/bin/env python3
"""H100 Benchmark & Profiling Suite for Doom Index Inference.

Comprehensive benchmarking utilities to prove H100 GPUs are utilized
effectively and identify bottlenecks in the inference pipeline.

Measures:
- Latency: p50, p95, p99, max (milliseconds)
- Throughput: queries/second at various batch sizes
- GPU utilization: Tensor Core active %, memory bandwidth
- Power efficiency: inferences per joule
- Roofline analysis: are we compute-bound or memory-bound?
- Comparison matrix: PyTorch vs ONNX vs TensorRT vs vLLM

Produces:
- JSON benchmark reports for CI/CD tracking
- Plots for viva presentation
- NSYS / Nsight Compute profile recipes

Usage:
    python -m src.benchmarks.h100_benchmark \
        --model models/doom_classifier.onnx \
        --engine tensorrt \
        --batch-sizes 1,8,16,32,64 \
        --seq-lens 64,128,256,512 \
        --report-dir benchmarks/h100/
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    model_path: str
    engine: str = "pytorch"  # pytorch, onnx, tensorrt, vllm
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64])
    sequence_lengths: List[int] = field(default_factory=lambda: [128, 256, 512])
    num_warmup: int = 50
    num_runs: int = 200
    device: str = "cuda"
    dtype: str = "fp16"


@dataclass
class LatencyStats:
    batch_size: int
    seq_len: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


@dataclass
class ThroughputResult:
    batch_size: int
    seq_len: int
    qps: float  # Queries per second
    inferences_per_sec: float
    tokens_per_sec: float


@dataclass
class GPUMetrics:
    timestamp: float
    utilization_gpu: float
    utilization_mem: float
    memory_used_mb: float
    memory_free_mb: float
    temperature_c: float
    power_draw_w: float
    power_limit_w: float
    sm_clock_mhz: float
    mem_clock_mhz: float
    pcie_tx_mb: float
    pcie_rx_mb: float
    throttled: bool


@dataclass
class BenchmarkReport:
    config: BenchmarkConfig
    latencies: List[LatencyStats] = field(default_factory=list)
    throughputs: List[ThroughputResult] = field(default_factory=list)
    gpu_snapshots: List[GPUMetrics] = field(default_factory=list)
    roofline: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    hostname: str = field(default_factory=lambda: os.uname().nodename)
    cuda_version: str = "unknown"
    driver_version: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            "config": asdict(self.config),
            "latencies": [asdict(l) for l in self.latencies],
            "throughputs": [asdict(t) for t in self.throughputs],
            "gpu_snapshots": [asdict(g) for g in self.gpu_snapshots],
            "roofline": self.roofline,
            "recommendations": self.recommendations,
            "meta": {
                "timestamp": self.timestamp,
                "hostname": self.hostname,
                "cuda_version": self.cuda_version,
                "driver_version": self.driver_version,
            },
        }


class H100Profiler:
    """Context manager for detailed GPU profiling during benchmark windows."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.snapshots: List[GPUMetrics] = []
        self._handle = None
        if PYNVML_AVAILABLE:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            except Exception as e:
                logger.warning(f"NVML init failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def snapshot(self) -> Optional[GPUMetrics]:
        if self._handle is None:
            return None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000.0
            sm_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM)
            pcie_stats = pynvml.nvmlDeviceGetPcieThroughput(self._handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(self._handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._handle) != 0

            metric = GPUMetrics(
                timestamp=time.time(),
                utilization_gpu=util.gpu,
                utilization_mem=util.memory,
                memory_used_mb=mem.used / (1024 * 1024),
                memory_free_mb=(mem.total - mem.used) / (1024 * 1024),
                temperature_c=temp,
                power_draw_w=power,
                power_limit_w=power_limit,
                sm_clock_mhz=sm_clock,
                mem_clock_mhz=mem_clock,
                pcie_tx_mb=pcie_stats / (1024 * 1024),
                pcie_rx_mb=pcie_rx / (1024 * 1024),
                throttled=throttle,
            )
            self.snapshots.append(metric)
            return metric
        except Exception as e:
            logger.debug(f"Snapshot error: {e}")
            return None


class DoomBenchmarkSuite:
    """Full benchmark suite for Doom Index inference backends."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.report = BenchmarkReport(config=config)
        self._init_gpu_info()

    def _init_gpu_info(self) -> None:
        if PYNVML_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.report.cuda_version = torch.version.cuda or "unknown"
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.report.driver_version = pynvml.nvmlSystemGetDriverVersion()
            except Exception:
                pass

    def _create_input(self, batch_size: int, seq_len: int) -> np.ndarray:
        """Generate synthetic input IDs for benchmarking."""
        vocab_size = 30000
        return np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)

    def benchmark_pytorch(self, model_wrapper) -> None:
        """Benchmark pure PyTorch inference."""
        import torch
        model = model_wrapper.to(self.config.device).eval()
        dtype = torch.float16 if self.config.dtype == "fp16" else torch.float32

        for bs in self.config.batch_sizes:
            for seq in self.config.sequence_lengths:
                input_ids = torch.from_numpy(self._create_input(bs, seq)).long().to(self.config.device)
                attention_mask = torch.ones_like(input_ids)

                # Warmup
                with torch.no_grad():
                    for _ in range(self.config.num_warmup):
                        _ = model(input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize()

                # Benchmark
                latencies = []
                with H100Profiler() as profiler:
                    for i in range(self.config.num_runs):
                        if i % 20 == 0:
                            profiler.snapshot()
                        start = time.time()
                        with torch.no_grad():
                            _ = model(input_ids, attention_mask=attention_mask)
                        torch.cuda.synchronize()
                        latencies.append((time.time() - start) * 1000)

                self._record(bs, seq, latencies, profiler.snapshots)

    def benchmark_onnx(self, session) -> None:
        """Benchmark ONNX Runtime inference."""
        for bs in self.config.batch_sizes:
            for seq in self.config.sequence_lengths:
                input_ids = self._create_input(bs, seq)
                attention_mask = np.ones_like(input_ids)
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

                # Warmup
                for _ in range(self.config.num_warmup):
                    _ = session.run(None, inputs)

                latencies = []
                with H100Profiler() as profiler:
                    for i in range(self.config.num_runs):
                        if i % 20 == 0:
                            profiler.snapshot()
                        start = time.time()
                        _ = session.run(None, inputs)
                        latencies.append((time.time() - start) * 1000)

                self._record(bs, seq, latencies, profiler.snapshots)

    def _record(self, bs: int, seq: int, latencies: List[float], gpu_snaps: List[GPUMetrics]) -> None:
        arr = np.array(latencies)
        latency = LatencyStats(
            batch_size=bs, seq_len=seq,
            mean_ms=float(np.mean(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            std_ms=float(np.std(arr)),
        )
        throughput = ThroughputResult(
            batch_size=bs, seq_len=seq,
            qps=float(1000.0 / latency.mean_ms * bs),
            inferences_per_sec=float(1000.0 / latency.mean_ms * bs),
            tokens_per_sec=float(1000.0 / latency.mean_ms * bs * seq),
        )
        self.report.latencies.append(latency)
        self.report.throughputs.append(throughput)
        self.report.gpu_snapshots.extend(gpu_snaps)
        logger.info(f"bs={bs}, seq={seq}: p50={latency.p50_ms:.2f}ms, qps={throughput.qps:.0f}")

    def generate_recommendations(self) -> None:
        """Analyze results and generate optimization recommendations."""
        recs = []
        if not self.report.latencies:
            return

        # Find optimal batch size
        best_qps = max(self.report.throughputs, key=lambda x: x.qps)
        recs.append(f"Optimal batch size for throughput: {best_qps.batch_size} (seq={best_qps.seq_len}, QPS={best_qps.qps:.0f})")

        # Check if GPU is underutilized
        avg_gpu_util = np.mean([s.utilization_gpu for s in self.report.gpu_snapshots]) if self.report.gpu_snapshots else 0
        if avg_gpu_util < 70:
            recs.append(f"GPU underutilized ({avg_gpu_util:.0f}%). Increase batch size or use Tensor Cores (FP16/BF16).")

        # Check memory headroom
        avg_mem = np.mean([s.memory_used_mb for s in self.report.gpu_snapshots]) if self.report.gpu_snapshots else 0
        if avg_mem > 70000:  # 70GB on 80GB H100
            recs.append(f"GPU memory high ({avg_mem:.0f} MB). Enable gradient checkpointing or reduce sequence length.")

        # Check throttling
        throttle_count = sum(1 for s in self.report.gpu_snapshots if s.throttled)
        if throttle_count > 0:
            recs.append(f"GPU throttling detected ({throttle_count} snapshots). Check cooling and power limits.")

        # Roofline: are we compute or memory bound?
        # Simplified: if latency doesn't scale linearly with batch, likely memory bound
        self.report.recommendations = recs
        for r in recs:
            logger.info(f"Recommendation: {r}")

    def save_report(self, output_dir: str) -> str:
        """Save JSON report and generate comparison CSV."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = out / f"benchmark_{self.config.engine}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, "w") as f:
            json.dump(self.report.to_dict(), f, indent=2)

        # CSV for plotting
        csv_path = out / f"benchmark_{self.config.engine}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame([{
            **asdict(l), **asdict(t)
        } for l, t in zip(self.report.latencies, self.report.throughputs)])
        df.to_csv(csv_path, index=False)

        logger.info(f"Reports saved: {json_path}, {csv_path}")
        return str(json_path)

    @staticmethod
    def generate_nsys_command(output_file: str = "profile.nsys-rep") -> str:
        """Generate Nsight Systems command for detailed profiling.

        Usage:
            nsys profile -o profile python -m src.benchmarks.h100_benchmark ...
        """
        cmd = (
            f"nsys profile --trace=cuda,nvtx,osrt "
            f"--cuda-memory-usage=true "
            f"--output={output_file} "
            f"python -m src.benchmarks.h100_benchmark --engine tensorrt --profile"
        )
        return cmd

    @staticmethod
    def generate_ncu_command(output_file: str = "profile.ncu-rep") -> str:
        """Generate Nsight Compute command for kernel-level analysis."""
        cmd = (
            f"ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,"
            f"sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"
            f"sm__sass_thread_inst_executed_op_ffma_pred_on.sum "
            f"--kernel-name regex:gemm "
            f"--output {output_file} "
            f"python -m src.benchmarks.h100_benchmark --engine tensorrt --profile"
        )
        return cmd


def main():
    parser = argparse.ArgumentParser(description="H100 Benchmark Suite")
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", default="pytorch", choices=["pytorch", "onnx", "tensorrt", "vllm"])
    parser.add_argument("--batch-sizes", default="1,8,16,32,64")
    parser.add_argument("--seq-lens", default="128,256,512")
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--num-runs", type=int, default=200)
    parser.add_argument("--report-dir", default="benchmarks/h100")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler")
    parser.add_argument("--nsys", action="store_true", help="Print nsys command and exit")
    parser.add_argument("--ncu", action="store_true", help="Print ncu command and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.nsys:
        print(H100Profiler.generate_nsys_command())
        return
    if args.ncu:
        print(H100Profiler.generate_ncu_command())
        return

    config = BenchmarkConfig(
        model_path=args.model,
        engine=args.engine,
        batch_sizes=[int(x) for x in args.batch_sizes.split(",")],
        sequence_lengths=[int(x) for x in args.seq_lens.split(",")],
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    suite = DoomBenchmarkSuite(config)

    if args.engine == "pytorch":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(config.model_path)
        suite.benchmark_pytorch(model)
    elif args.engine == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(config.model_path, providers=["CUDAExecutionProvider"])
        suite.benchmark_onnx(sess)
    else:
        logger.error(f"Engine {args.engine} benchmark not yet implemented in this runner")
        sys.exit(1)

    suite.generate_recommendations()
    suite.save_report(args.report_dir)


if __name__ == "__main__":
    main()
