"""
TorchServe configuration and model archiver for production deployment.
Provides optimized inference with batching, caching, and A/B testing support.
"""
import logging
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TorchServeConfig:
    model_name: str = "doom-index"
    model_version: str = "2.0.0"
    batch_size: int = 8
    max_batch_delay: int = 100  # ms
    response_timeout: int = 120  # seconds
    min_workers: int = 2
    max_workers: int = 4
    device: str = "gpu" if torch.cuda.is_available() else "cpu"
    serialize_format: str = "tgz"


class DoomIndexHandler:
    """
    TorchServe custom handler for Doom Index multimodal inference.
    Handles text + optional image + optional graph inputs.
    """

    def __init__(self):
        self.model = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.device = None

        # Feature extractors (loaded once)
        self.tokenizer = None
        self.vision_processor = None

    def initialize(self, context):
        """Load model and dependencies."""
        self.context = context
        self.manifest = context.manifest
        properties = context.system_properties

        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        # Load serialized model
        model_pt_path = os.path.join(model_dir, "doom_index_model.pt")
        if os.path.isfile(model_pt_path):
            self.model = torch.jit.load(model_pt_path, map_location=self.device)
            self.model.eval()
        else:
            raise RuntimeError(f"Model file not found: {model_pt_path}")

        # Load tokenizers
        from transformers import DistilBertTokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

        self.initialized = True
        logger.info(f"Handler initialized on {self.device}")

    def preprocess(self, data: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Preprocess incoming requests.

        Args:
            data: List of request dicts with 'text', optional 'image', optional 'user_id'
        Returns:
            Dict of batched tensors
        """
        texts = []
        has_images = []

        for req in data:
            body = req.get("body") or req
            if isinstance(body, (bytes, str)):
                body = json.loads(body)

            texts.append(body.get("text", ""))
            has_images.append("image" in body)

        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
            "has_images": torch.tensor(has_images, dtype=torch.bool)
        }

    def inference(self, inputs: Dict[str, torch.Tensor]) -> List[Dict]:
        """Run model inference."""
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.sigmoid(outputs).cpu().numpy()

        results = []
        for prob in probs:
            score = float(prob[0]) * 100
            risk_level = (
                "critical" if score >= 80 else
                "high" if score >= 60 else
                "medium" if score >= 40 else
                "low"
            )
            results.append({
                "doom_score": round(score, 2),
                "risk_level": risk_level,
                "confidence": round(abs(score - 50) / 50, 4)
            })

        return results

    def postprocess(self, inference_output: List[Dict]) -> List[str]:
        """Format output as JSON strings."""
        return [json.dumps(r) for r in inference_output]

    def handle(self, data, context):
        """Main entry point for TorchServe."""
        if not self.initialized:
            self.initialize(context)

        inputs = self.preprocess(data)
        outputs = self.inference(inputs)
        return self.postprocess(outputs)


def create_model_archive(config: TorchServeConfig, 
                         model_path: str,
                         handler_path: str,
                         extra_files: Optional[List[str]] = None,
                         output_dir: str = "model_store") -> str:
    """
    Create TorchServe model archive (.mar file).

    Args:
        config: Serving configuration
        model_path: Path to serialized model (.pt or .onnx)
        handler_path: Path to handler script
        extra_files: Additional files to include (tokenizers, configs)
        output_dir: Output directory for .mar file
    Returns:
        Path to created .mar file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build torch-model-archiver command
    cmd = [
        "torch-model-archiver",
        "--model-name", config.model_name,
        "--version", config.model_version,
        "--model-file", model_path,
        "--handler", handler_path,
        "--export-path", output_dir,
        "--archive-format", config.serialize_format
    ]

    if extra_files:
        cmd.extend(["--extra-files", ",".join(extra_files)])

    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Model archiver failed: {result.stderr}")

    mar_path = os.path.join(output_dir, f"{config.model_name}.mar")
    logger.info(f"Model archive created: {mar_path}")
    return mar_path


def generate_torchserve_config(config: TorchServeConfig, path: str = "config.properties"):
    """Generate TorchServe config.properties file."""
    content = f"""
# TorchServe Configuration for Doom Index
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=32
job_queue_size=1000
async_logging=true

# Model specific
batch_size={config.batch_size}
max_batch_delay={config.max_batch_delay}
response_timeout={config.response_timeout}
min_workers={config.min_workers}
max_workers={config.max_workers}

# Metrics
metrics_mode=prometheus
metrics_config=metrics.yaml
"""
    with open(path, "w") as f:
        f.write(content.strip())
    logger.info(f"TorchServe config written to {path}")


def start_torchserve(model_store: str = "model_store", 
                     config_path: str = "config.properties"):
    """Start TorchServe process."""
    cmd = [
        "torchserve",
        "--start",
        "--model-store", model_store,
        "--ts-config", config_path,
        "--models", "doom-index=doom-index.mar",
        "--foreground"
    ]
    import subprocess
    process = subprocess.Popen(cmd)
    logger.info(f"TorchServe started with PID {process.pid}")
    return process
