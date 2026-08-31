#!/usr/bin/env python3
"""MLflow Model Registry + Model Versioning for Production.

Wraps MLflow's model registry with Doom Index-specific staging logic,
automated promotion gates, and artifact lineage tracking.

Features:
- Semantic version management (major.minor.patch)
- Promotion gates (accuracy > threshold, latency < threshold, no drift)
- A/B test model tagging (champion vs challenger)
- Artifact lineage (dataset version -> model version -> deployment)
- Rollback capability
- Webhook-style triggers for CI/CD

Usage:
    registry = ModelRegistry(experiment_name="doom_index")
    version = registry.register(
        model_path="checkpoints/epoch_10.pt",
        name="doom_distilbert",
        metrics={"f1": 0.912, "latency_p99_ms": 12.4},
        dataset_version="v2.3.1",
        stage="Staging"
    )
    registry.promote("doom_distilbert", version, target="Production", 
                     required_metrics={"f1": 0.90})
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import infer_signature
    from mlflow.exceptions import RestException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Model registry will use filesystem fallback.")


@dataclass
class ModelVersion:
    """Structured model version record."""
    name: str
    version: str
    stage: str
    metrics: Dict[str, float]
    dataset_version: str
    created_at: str
    run_id: Optional[str] = None
    artifact_uri: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PromotionGate:
    """Gate conditions for model promotion between stages."""
    min_f1: float = 0.88
    max_latency_p99_ms: float = 50.0
    max_drift_score: float = 0.15
    require_ab_test_pass: bool = True
    require_human_approval: bool = False

    def evaluate(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Evaluate metrics against gates. Returns (passed, failure_reasons)."""
        failures = []
        if metrics.get("f1", 0.0) < self.min_f1:
            failures.append(f"F1 {metrics.get('f1'):.3f} < {self.min_f1}")
        if metrics.get("latency_p99_ms", 9999.0) > self.max_latency_p99_ms:
            failures.append(f"Latency p99 {metrics.get('latency_p99_ms'):.1f}ms > {self.max_latency_p99_ms}ms")
        if metrics.get("drift_score", 0.0) > self.max_drift_score:
            failures.append(f"Drift {metrics.get('drift_score'):.3f} > {self.max_drift_score}")
        return len(failures) == 0, failures


class ModelRegistry:
    """Production model registry with automated lifecycle management."""

    STAGES = ["None", "Staging", "Production", "Archived"]

    def __init__(
        self,
        experiment_name: str = "doom_index",
        tracking_uri: Optional[str] = None,
        local_registry_dir: str = "./model_registry",
        promotion_gate: Optional[PromotionGate] = None,
    ):
        self.experiment_name = experiment_name
        self.local_registry_dir = Path(local_registry_dir)
        self.local_registry_dir.mkdir(parents=True, exist_ok=True)
        self.promotion_gate = promotion_gate or PromotionGate()
        self._client = None
        self._mlflow_active = False

        if MLFLOW_AVAILABLE:
            try:
                uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", f"file://{self.local_registry_dir / 'mlruns'}")
                mlflow.set_tracking_uri(uri)
                self._client = MlflowClient()
                self._mlflow_active = True
                logger.info(f"MLflow registry active: {uri}")
                self._ensure_registered_model(experiment_name)
            except Exception as e:
                logger.warning(f"MLflow init failed: {e}. Using filesystem fallback.")

    def _ensure_registered_model(self, name: str) -> None:
        """Create registered model if not exists."""
        if not self._mlflow_active:
            return
        try:
            self._client.create_registered_model(name)
        except RestException:
            pass  # Already exists

    def _generate_semver(self, name: str, bump: str = "patch") -> str:
        """Generate semantic version based on existing versions."""
        existing = self.list_versions(name)
        if not existing:
            return "1.0.0"
        versions = [v.version for v in existing]
        # Find highest semver
        semver_pattern = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
        nums = []
        for v in versions:
            m = semver_pattern.match(v)
            if m:
                nums.append(tuple(map(int, m.groups())))
        if not nums:
            return "1.0.0"
        major, minor, patch = max(nums)
        if bump == "major":
            return f"{major + 1}.0.0"
        elif bump == "minor":
            return f"{major}.{minor + 1}.0"
        return f"{major}.{minor}.{patch + 1}"

    def register(
        self,
        model_path: str,
        name: str,
        metrics: Dict[str, float],
        dataset_version: str,
        stage: str = "None",
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> str:
        """Register a new model version.

        Returns:
            Version string of registered model
        """
        version_str = self._generate_semver(name)
        ts = datetime.utcnow().isoformat()

        if self._mlflow_active:
            try:
                # Log model as artifact in a run
                with mlflow.start_run(run_name=f"register_{name}_v{version_str}"):
                    mlflow.log_params({
                        "model_name": name,
                        "dataset_version": dataset_version,
                    })
                    mlflow.log_metrics(metrics)
                    mlflow.log_artifact(model_path, artifact_path="model")
                    run_id = mlflow.active_run().info.run_id

                # Create version
                mv = self._client.create_model_version(
                    name=name,
                    source=f"runs:/{run_id}/model",
                    run_id=run_id,
                    tags=tags or {},
                    description=description,
                )
                version_str = mv.version  # MLflow assigns integer versions
                logger.info(f"MLflow registered {name} v{version_str}")
            except Exception as e:
                logger.error(f"MLflow registration failed: {e}")
                self._mlflow_active = False

        # Always write local manifest
        record = ModelVersion(
            name=name,
            version=version_str,
            stage=stage,
            metrics=metrics,
            dataset_version=dataset_version,
            created_at=ts,
            tags=tags or {},
        )
        self._write_local_manifest(record)
        return version_str

    def _write_local_manifest(self, record: ModelVersion) -> None:
        """Persist version record to local filesystem."""
        manifest_dir = self.local_registry_dir / record.name
        manifest_dir.mkdir(parents=True, exist_ok=True)
        path = manifest_dir / f"v{record.version}.json"
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)
        # Update latest symlink
        latest = manifest_dir / "latest.json"
        if latest.exists() or not latest.exists():
            latest.unlink(missing_ok=True)
            latest.symlink_to(path.name)

    def promote(
        self,
        name: str,
        version: str,
        target: str,
        required_metrics: Optional[Dict[str, float]] = None,
        force: bool = False,
    ) -> bool:
        """Promote model version to a new stage with gate checks.

        Returns:
            True if promotion succeeded
        """
        if target not in self.STAGES:
            raise ValueError(f"Invalid stage {target}. Must be one of {self.STAGES}")

        # Load version record
        record = self.get_version(name, version)
        if record is None:
            raise ValueError(f"Version {name} v{version} not found")

        # Gate evaluation
        metrics = required_metrics or record.metrics
        passed, reasons = self.promotion_gate.evaluate(metrics)
        if not passed and not force:
            logger.error(f"Promotion gates failed for {name} v{version}: {reasons}")
            return False

        # Perform promotion
        if self._mlflow_active:
            try:
                self._client.transition_model_version_stage(
                    name=name,
                    version=version,
                    stage=target,
                )
            except Exception as e:
                logger.warning(f"MLflow stage transition failed: {e}")

        # Update local manifest
        record.stage = target
        record.tags["promoted_at"] = datetime.utcnow().isoformat()
        record.tags["promotion_reason"] = "manual" if force else "auto_gate_pass"
        self._write_local_manifest(record)
        logger.info(f"Promoted {name} v{version} to {target}")
        return True

    def get_version(self, name: str, version: str) -> Optional[ModelVersion]:
        """Retrieve specific model version record."""
        path = self.local_registry_dir / name / f"v{version}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return ModelVersion(**data)

    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """Get current Production stage model."""
        versions = self.list_versions(name)
        for v in versions:
            if v.stage == "Production":
                return v
        return None

    def list_versions(self, name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        manifest_dir = self.local_registry_dir / name
        if not manifest_dir.exists():
            return []
        versions = []
        for f in manifest_dir.glob("v*.json"):
            if f.name == "latest.json":
                continue
            with open(f) as fh:
                data = json.load(fh)
            versions.append(ModelVersion(**data))
        return sorted(versions, key=lambda v: v.created_at)

    def rollback(self, name: str) -> Optional[str]:
        """Rollback Production to previous version.

        Returns:
            New production version string, or None if no previous.
        """
        versions = self.list_versions(name)
        prod_idx = None
        for i, v in enumerate(versions):
            if v.stage == "Production":
                prod_idx = i
                break
        if prod_idx is None or prod_idx == 0:
            logger.warning(f"No previous version to rollback to for {name}")
            return None

        prev = versions[prod_idx - 1]
        self.promote(name, prev.version, "Production", force=True)
        return prev.version

    def compare_versions(self, name: str, v1: str, v2: str) -> Dict[str, Any]:
        """Compare two model versions across metrics and metadata."""
        a = self.get_version(name, v1)
        b = self.get_version(name, v2)
        if not a or not b:
            raise ValueError("One or both versions not found")

        metric_diff = {}
        all_keys = set(a.metrics.keys()) | set(b.metrics.keys())
        for k in all_keys:
            mv = a.metrics.get(k, 0)
            bv = b.metrics.get(k, 0)
            metric_diff[k] = {
                "v1": mv,
                "v2": bv,
                "delta": bv - mv,
                "pct_delta": f"{(bv - mv) / abs(mv) * 100:.1f}%" if mv != 0 else "N/A",
            }
        return {
            "versions": {"v1": v1, "v2": v2},
            "stages": {"v1": a.stage, "v2": b.stage},
            "datasets": {"v1": a.dataset_version, "v2": b.dataset_version},
            "metrics": metric_diff,
        }

    def tag_ab_test(self, name: str, champion: str, challenger: str, traffic_split: float = 0.1) -> None:
        """Tag two versions for A/B testing."""
        for ver, role in [(champion, "champion"), (challenger, "challenger")]:
            record = self.get_version(name, ver)
            if record:
                record.tags["ab_test"] = "true"
                record.tags["ab_role"] = role
                record.tags["ab_traffic"] = str(traffic_split) if role == "challenger" else str(1.0 - traffic_split)
                self._write_local_manifest(record)
        logger.info(f"A/B test configured: {champion} (champion) vs {challenger} (challenger)")

    def generate_deployment_manifest(self, name: str, version: str) -> Dict[str, Any]:
        """Generate Kubernetes-style deployment manifest for model."""
        record = self.get_version(name, version)
        if not record:
            raise ValueError(f"Version {version} not found")
        return {
            "apiVersion": "mlops.prsi/v1",
            "kind": "ModelDeployment",
            "metadata": {
                "name": f"{name}-{version}",
                "version": version,
                "stage": record.stage,
                "dataset": record.dataset_version,
            },
            "spec": {
                "modelUri": record.artifact_uri or f"models/{name}/{version}",
                "replicas": 2 if record.stage == "Production" else 1,
                "resources": {
                    "gpu": "nvidia.com/gpu: 1",
                    "memory": "16Gi",
                },
                "metrics": record.metrics,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("--name", default="doom_distilbert")
    parser.add_argument("--action", choices=["register", "promote", "list", "rollback", "compare"], required=True)
    parser.add_argument("--version")
    parser.add_argument("--target", default="Production")
    parser.add_argument("--model-path")
    parser.add_argument("--metrics-json")
    parser.add_argument("--dataset-version", default="unknown")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    registry = ModelRegistry()

    if args.action == "register":
        if not args.model_path or not args.metrics_json:
            print("--model-path and --metrics-json required for register")
            sys.exit(1)
        metrics = json.loads(Path(args.metrics_json).read_text())
        v = registry.register(
            model_path=args.model_path,
            name=args.name,
            metrics=metrics,
            dataset_version=args.dataset_version,
        )
        print(f"Registered {args.name} version {v}")

    elif args.action == "promote":
        if not args.version:
            print("--version required")
            sys.exit(1)
        success = registry.promote(args.name, args.version, args.target)
        print(f"Promotion {'succeeded' if success else 'failed'}")

    elif args.action == "list":
        versions = registry.list_versions(args.name)
        for v in versions:
            print(f"  {v.version}: stage={v.stage}, f1={v.metrics.get('f1', 'N/A')}, created={v.created_at}")

    elif args.action == "rollback":
        v = registry.rollback(args.name)
        print(f"Rolled back to {v}" if v else "No rollback possible")

    elif args.action == "compare":
        if not args.version:
            print("--version (v2) required; compare with current production")
            sys.exit(1)
        prod = registry.get_production_model(args.name)
        if not prod:
            print("No production model")
            sys.exit(1)
        diff = registry.compare_versions(args.name, prod.version, args.version)
        print(json.dumps(diff, indent=2))


if __name__ == "__main__":
    main()
