"""
DVC pipeline utilities for programmatic pipeline execution and monitoring.
"""
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DVCPipelineConfig:
    dvc_file: str = "dvc.yaml"
    metrics_dir: str = "metrics"
    reports_dir: str = "reports"
    remote_storage: str = "gs://doom-index-dvc"  # or s3://, azure://


class DVCPipelineManager:
    """
    Manage DVC pipelines programmatically.
    Handles stage execution, metrics tracking, and artifact versioning.
    """

    def __init__(self, config: Optional[DVCPipelineConfig] = None):
        self.config = config or DVCPipelineConfig()
        Path(self.config.metrics_dir).mkdir(exist_ok=True)
        Path(self.config.reports_dir).mkdir(exist_ok=True)
        logger.info("DVCPipelineManager initialized")

    def run_stage(self, stage_name: str, force: bool = False) -> Dict[str, Any]:
        """
        Run a specific DVC stage.

        Args:
            stage_name: Name of stage in dvc.yaml
            force: Force rerun even if no changes
        Returns:
            Dict with returncode, stdout, stderr
        """
        cmd = ["dvc", "repro", stage_name]
        if force:
            cmd.append("--force")

        logger.info(f"Running DVC stage: {stage_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Stage {stage_name} failed: {result.stderr}")
        else:
            logger.info(f"Stage {stage_name} completed successfully")

        return {
            "stage": stage_name,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    def run_full_pipeline(self, force: bool = False) -> List[Dict[str, Any]]:
        """Run complete pipeline from scratch."""
        stages = [
            "prepare",
            "build_graph", 
            "extract_features",
            "train_text",
            "train_graph",
            "train_multimodal",
            "adversarial_training",
            "evaluate",
            "export_onnx"
        ]

        results = []
        for stage in stages:
            result = self.run_stage(stage, force=force)
            results.append(result)

            if result["returncode"] != 0:
                logger.error(f"Pipeline halted at stage: {stage}")
                break

        return results

    def get_metrics(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """Fetch metrics from DVC."""
        cmd = ["dvc", "metrics", "show"]
        if stage:
            cmd.extend(["--target", f"metrics/{stage}.json"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"raw": result.stdout}
        return {"error": result.stderr}

    def compare_metrics(self, rev1: str = "HEAD", rev2: str = "HEAD~1") -> Dict[str, Any]:
        """Compare metrics between Git commits."""
        cmd = ["dvc", "metrics", "diff", rev1, rev2]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"raw": result.stdout}
        return {"error": result.stderr}

    def push_artifacts(self):
        """Push DVC-tracked artifacts to remote storage."""
        cmd = ["dvc", "push"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Artifacts pushed to remote storage")
        else:
            logger.error(f"Push failed: {result.stderr}")

        return result.returncode == 0

    def pull_artifacts(self):
        """Pull DVC-tracked artifacts from remote storage."""
        cmd = ["dvc", "pull"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Artifacts pulled from remote storage")
        else:
            logger.error(f"Pull failed: {result.stderr}")

        return result.returncode == 0

    def get_pipeline_status(self) -> Dict[str, str]:
        """Check which stages are up-to-date."""
        cmd = ["dvc", "status"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            if not result.stdout.strip():
                return {"status": "up_to_date", "details": {}}
            try:
                return {"status": "outdated", "details": json.loads(result.stdout)}
            except json.JSONDecodeError:
                return {"status": "unknown", "details": result.stdout}

        return {"status": "error", "details": result.stderr}

    def tag_model_version(self, tag: str, message: str = ""):
        """Tag current model state with Git + DVC."""
        # Git tag
        git_cmd = ["git", "tag", "-a", tag, "-m", message or f"Model version {tag}"]
        subprocess.run(git_cmd, check=True)

        # DVC tag
        dvc_cmd = ["dvc", "params", "diff", f"--rev", tag]
        subprocess.run(dvc_cmd)

        logger.info(f"Tagged model version: {tag}")
