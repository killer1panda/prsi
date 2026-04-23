#!/usr/bin/env python3
"""Validate Doom Index v2 setup before training.

Checks:
- Python version
- CUDA availability and version
- Required packages
- Data files
- Neo4j connectivity
- GPU memory

Usage:
    python validate_setup.py
"""

import importlib
import sys
from pathlib import Path


def check_python():
    """Check Python version."""
    print("Python Version:")
    print(f"  {sys.version}")

    if sys.version_info < (3, 10):
        print("  [FAIL] Python 3.10+ required")
        return False
    print("  [PASS] Python 3.10+")
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\nCUDA:")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [PASS] CUDA available")
            print(f"  Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
                print(f"    Compute capability: {props.major}.{props.minor}")
                if props.major < 7:
                    print(f"    [WARN] Compute capability < 7.0 may have issues")
            return True
        else:
            print("  [WARN] CUDA not available (CPU training only)")
            return True  # Not a hard failure
    except ImportError:
        print("  [FAIL] PyTorch not installed")
        return False


def check_packages():
    """Check required packages."""
    print("\nRequired Packages:")

    required = {
        "torch": "PyTorch",
        "torch_geometric": "PyTorch Geometric",
        "transformers": "HuggingFace Transformers",
        "sklearn": "scikit-learn",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "fastapi": "FastAPI",
        "pymongo": "PyMongo",
        "neo4j": "Neo4j Driver",
        "tqdm": "tqdm",
    }

    all_ok = True
    for module, name in required.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [PASS] {name:25s} {version}")
        except ImportError:
            print(f"  [FAIL] {name:25s} NOT INSTALLED")
            all_ok = False

    return all_ok


def check_data():
    """Check data files."""
    print("\nData Files:")

    paths = [
        "data/processed_reddit_multimodal.csv",
        "processed_sample.csv",
    ]

    found = False
    for p in paths:
        path = Path(p)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [PASS] {p} ({size_mb:.1f} MB)")
            found = True

            # Check columns
            import pandas as pd
            df = pd.read_csv(p, nrows=5)
            required_cols = ['text', 'author_id', 'label']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"    [FAIL] Missing columns: {missing}")
            else:
                print(f"    [PASS] Required columns present")

    if not found:
        print("  [FAIL] No processed data found")
        print("  Run: python train_model_full_fixed.py")
        return False

    return True


def check_neo4j():
    """Check Neo4j connectivity."""
    print("\nNeo4j:")

    try:
        from src.data.neo4j_connector import get_neo4j
        neo4j = get_neo4j()

        with neo4j.driver.session(database=neo4j.database) as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                print("  [PASS] Neo4j connected")

                # Check node counts
                counts = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                print(f"  Nodes in graph: {counts}")
                return True
    except Exception as e:
        print(f"  [WARN] Neo4j not available: {e}")
        print("  Start with: docker-compose up neo4j -d")
        return True  # Not a hard failure


def check_models():
    """Check model files."""
    print("\nModel Files:")

    model_path = Path("models/multimodal_doom/best_model.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  [PASS] Trained model found ({size_mb:.1f} MB)")
        return True
    else:
        print("  [INFO] No trained model found (will be created during training)")
        return True


def check_disk_space():
    """Check available disk space."""
    print("\nDisk Space:")

    import shutil
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    print(f"  Free space: {free_gb:.1f} GB")

    if free_gb < 10:
        print("  [WARN] Less than 10 GB free")
        return False
    print("  [PASS] Sufficient disk space")
    return True


def main():
    print("=" * 60)
    print("Doom Index v2 — Setup Validation")
    print("=" * 60)

    checks = [
        ("Python", check_python),
        ("CUDA", check_cuda),
        ("Packages", check_packages),
        ("Data", check_data),
        ("Neo4j", check_neo4j),
        ("Models", check_models),
        ("Disk", check_disk_space),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("Summary:")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:6s} {name}")
    print("=" * 60)

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All checks passed! Ready to train.")
    else:
        print("\n⚠ Some checks failed. Fix issues before training.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
