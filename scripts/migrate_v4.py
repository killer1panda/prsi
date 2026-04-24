#!/usr/bin/env python3
"""
Migration script v4: Places all generated files into correct directory structure.
Run this after extracting the zip into your prsi/ repo root.

Usage:
    python scripts/migrate_v4.py --force
"""
import argparse
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class FileMapping:
    source: str
    target: str
    overwrite: bool = True


MIGRATIONS: List[FileMapping] = [
    # Core Models
    FileMapping("src/models/vision_encoder.py", "src/models/vision_encoder.py"),
    FileMapping("src/models/meme_detector.py", "src/models/meme_detector.py"),
    FileMapping("src/models/temporal_gnn.py", "src/models/temporal_gnn.py"),
    FileMapping("src/models/gat_model.py", "src/models/gat_model.py"),
    FileMapping("src/models/contrastive_pretrain.py", "src/models/contrastive_pretrain.py"),
    FileMapping("src/models/multilingual.py", "src/models/multilingual.py"),
    FileMapping("src/models/interpretability.py", "src/models/interpretability.py"),
    FileMapping("src/models/hyperparam_search.py", "src/models/hyperparam_search.py"),
    FileMapping("src/models/calibration.py", "src/models/calibration.py"),
    FileMapping("src/models/ensemble.py", "src/models/ensemble.py"),
    FileMapping("src/models/onnx_runtime.py", "src/models/onnx_runtime.py"),
    FileMapping("src/models/fusion.py", "src/models/fusion.py"),
    FileMapping("src/models/temporal.py", "src/models/temporal.py"),
    
    # Attacks
    FileMapping("src/attacks/adversarial_production.py", "src/attacks/adversarial_production.py"),
    FileMapping("src/attacks/adversarial_training.py", "src/attacks/adversarial_training.py"),
    
    # Data
    FileMapping("src/data/build_neo4j_graph_production.py", "src/data/build_neo4j_graph_production.py"),
    FileMapping("src/data/weak_labeling.py", "src/data/weak_labeling.py"),
    FileMapping("src/data/multimodal_dataset.py", "src/data/multimodal_dataset.py"),
    
    # Features
    FileMapping("src/features/feature_store.py", "src/features/feature_store.py"),
    
    # Streaming
    FileMapping("src/streaming/kafka_pipeline.py", "src/streaming/kafka_pipeline.py"),
    FileMapping("src/streaming/beam_pipeline.py", "src/streaming/beam_pipeline.py"),
    
    # API
    FileMapping("src/api/api_v2_production.py", "src/api/api_v2_production.py"),
    FileMapping("src/api/cache.py", "src/api/cache.py"),
    FileMapping("src/api/monitoring.py", "src/api/monitoring.py"),
    FileMapping("src/api/torchserve_config.py", "src/api/torchserve_config.py"),
    
    # Dashboard
    FileMapping("src/dashboard/app_production.py", "src/dashboard/app.py"),
    
    # Privacy
    FileMapping("src/privacy/dp_trainer.py", "src/privacy/dp_trainer.py"),
    FileMapping("src/privacy/fl_simulator.py", "src/privacy/fl_simulator.py"),
    
    # Evaluation
    FileMapping("src/evaluation/evaluate_full.py", "src/evaluation/evaluate_full.py"),
    FileMapping("src/evaluation/ab_testing.py", "src/evaluation/ab_testing.py"),
    FileMapping("src/models/drift_detector.py", "src/models/drift_detector.py"),
    FileMapping("src/models/fairness.py", "src/models/fairness.py"),
    
    # Training
    FileMapping("scripts/hpc_orchestrator.py", "scripts/hpc_orchestrator.py"),
    FileMapping("scripts/slurm_launcher.sh", "scripts/slurm_launcher.sh"),
    
    # Tests
    FileMapping("tests/test_integration.py", "tests/test_integration.py"),
    FileMapping("tests/test_api.py", "tests/test_api.py"),
    FileMapping("tests/test_load.py", "tests/test_load.py"),
    
    # CI/CD
    FileMapping(".github/workflows/ci.yml", ".github/workflows/ci.yml"),
    FileMapping(".github/workflows/cd.yml", ".github/workflows/cd.yml"),
    
    # Docker
    FileMapping("docker-compose-production.yml", "docker-compose.yml"),
    FileMapping("docker/Dockerfile.api", "docker/Dockerfile.api"),
    FileMapping("docker/Dockerfile.dashboard", "docker/Dockerfile.dashboard"),
    FileMapping("nginx/nginx.conf", "nginx/nginx.conf"),
    
    # Configs
    FileMapping("configs/hpc_distilbert.yaml", "configs/hpc_distilbert.yaml"),
    FileMapping("configs/multimodal.yaml", "configs/multimodal.yaml"),
    FileMapping("configs/eval.yaml", "configs/eval.yaml"),
    
    # Monitoring
    FileMapping("monitoring/prometheus.yml", "monitoring/prometheus.yml"),
    FileMapping("monitoring/grafana/datasources/prometheus.yml", "monitoring/grafana/datasources/prometheus.yml"),
    
    # Requirements
    FileMapping("requirements.txt", "requirements.txt"),
    FileMapping("requirements-api.txt", "requirements-api.txt"),
    FileMapping("requirements-dashboard.txt", "requirements-dashboard.txt"),
    FileMapping("requirements-dev.txt", "requirements-dev.txt"),
    
    # Root files
    FileMapping("Makefile", "Makefile"),
    FileMapping("dvc.yaml", "dvc.yaml"),
    FileMapping("docs/model_card.md", "docs/model_card.md"),
]


def migrate(force: bool = False):
    """Execute all file migrations."""
    print("=" * 60)
    print("DOOM INDEX V4 MIGRATION")
    print("=" * 60)
    
    success = 0
    skipped = 0
    failed = 0
    
    for mapping in MIGRATIONS:
        source = Path(mapping.source)
        target = Path(mapping.target)
        
        if not source.exists():
            print(f"  [MISSING] {mapping.source}")
            failed += 1
            continue
        
        target.parent.mkdir(parents=True, exist_ok=True)
        
        if target.exists() and not force:
            print(f"  [SKIP] {mapping.target} (exists, use --force to overwrite)")
            skipped += 1
            continue
        
        try:
            shutil.copy2(source, target)
            print(f"  [OK] {mapping.source} -> {mapping.target}")
            success += 1
        except Exception as e:
            print(f"  [FAIL] {mapping.source}: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Migration complete: {success} copied, {skipped} skipped, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\nSome files were missing. This is expected if you haven't extracted all zips.")
        print("Run this script again after extracting all files.")
    
    # Create necessary directories
    for dir_path in [
        "data/raw", "data/processed", "data/graph", "data/features",
        "models", "models/distilbert", "models/gnn", "models/multimodal", "models/robust",
        "logs", "reports", "checkpoints", "viva_plots",
        "src/__init__.py", "src/data/__init__.py", "src/models/__init__.py",
        "src/features/__init__.py", "src/attacks/__init__.py",
        "src/privacy/__init__.py", "src/api/__init__.py",
        "src/dashboard/__init__.py", "src/streaming/__init__.py",
        "src/evaluation/__init__.py"
    ]:
        path = Path(dir_path)
        if dir_path.endswith("__init__.py"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    
    print("\nDirectory structure created.")
    print("\nNext steps:")
    print("  1. make install-hpc")
    print("  2. make validate")
    print("  3. make data-prepare")
    print("  4. make graph-build")
    print("  5. make train-hpc")


def main():
    parser = argparse.ArgumentParser(description="Migrate Doom Index v4 files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    
    migrate(force=args.force)


if __name__ == "__main__":
    main()
