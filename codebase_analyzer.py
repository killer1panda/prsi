#!/usr/bin/env python3
"""
Codebase Analyzer for Doom Index - Identifies missing implementations and gaps
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Set
import ast
import json

class CodebaseAnalyzer:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.python_files: List[Path] = []
        self.missing_implementations: List[Dict] = []
        self.found_implementations: List[Dict] = []
        self.analysis_report: Dict = {}
        
    def scan_codebase(self) -> Dict:
        """Scan the entire codebase for implementation gaps"""
        print("Scanning codebase...")
        
        # Find all Python files
        self.python_files = list(self.root_path.rglob("*.py"))
        print(f"Found {len(self.python_files)} Python files")
        
        # Check for core components
        self._check_model_implementations()
        self._check_data_pipeline_implementations()
        self._check_api_implementations()
        self._check_training_implementations()
        self._check_deployment_implementations()
        
        return self.analysis_report
    
    def _check_model_implementations(self):
        """Check for model-related implementations"""
        model_files = [
            "src/models/predictor.py",
            "src/models/integrated_predictor.py",
            "src/models/gnn_model.py",
            "src/models/gat_model.py"
        ]
        
        missing = []
        found = []
        
        for file_path in model_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                found.append({
                    "file": file_path,
                    "status": "implemented",
                    "details": "Model component exists"
                })
            else:
                missing.append({
                    "file": file_path,
                    "status": "missing",
                    "details": "Model component not found"
                })
        
        self.analysis_report["models"] = {
            "found": found,
            "missing": missing
        }
    
    def _check_data_pipeline_implementations(self):
        """Check data pipeline components"""
        data_files = [
            "src/data/pipeline.py",
            "src/data/preprocessing.py",
            "src/data/scrapers/",
            "src/features/engineering.py"
        ]
        
        missing = []
        found = []
        
        for file_path in data_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                found.append({
                    "file": file_path,
                    "status": "implemented",
                    "details": "Data component exists"
                })
            else:
                missing.append({
                    "file": file_path,
                    "status": "missing",
                    "details": "Data component not implemented"
                })
        
        self.analysis_report["data_pipeline"] = {
            "found": found,
            "missing": missing
        }
    
    def _check_api_implementations(self):
        """Check API-related implementations"""
        api_files = [
            "src/api/api_v2_production.py",
            "src/api/monitoring.py",
            "src/api/cache.py"
        ]
        
        missing = []
        found = []
        
        for file_path in api_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                found.append({
                    "file": file_path,
                    "status": "implemented",
                    "details": "API component exists"
                })
            else:
                missing.append({
                    "file": file_path,
                    "status": "missing",
                    "details": "API component not implemented"
                })
        
        self.analysis_report["api"] = {
            "found": found,
            "missing": missing
        }
    
    def _check_training_implementations(self):
        """Check training-related implementations"""
        training_files = [
            "train_model.py",
            "train_model_full.py",
            "src/models/multimodal_trainer.py"
        ]
        
        missing = []
        found = []
        
        for file_path in training_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                found.append({
                    "file": file_path,
                    "status": "implemented",
                    "details": "Training component exists"
                })
            else:
                missing.append({
                    "file": file_path,
                    "status": "missing",
                    "details": "Training component not implemented"
                })
        
        self.analysis_report["training"] = {
            "found": found,
            "missing": missing
        }
    
    def _check_deployment_implementations(self):
        """Check deployment-related implementations"""
        deployment_files = [
            "src/api/torchserve_config.py",
            "src/inference/tensorrt_optimizer.py"
        ]
        
        missing = []
        found = []
        
        for file_path in deployment_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                found.append({
                    "file": file_path,
                    "status": "implemented",
                    "details": "Deployment component exists"
                })
            else:
                missing.append({
                    "file": file_path,
                    "status": "missing",
                    "details": "Deployment component not implemented"
                })
        
        self.analysis_report["deployment"] = {
            "found": found,
            "missing": missing
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of the analysis"""
        report = []
        report.append("DOOM INDEX CODEBASE ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Overall status
        total_files = len(self.python_files)
        report.append(f"Total Python files found: {total_files}")
        report.append("")
        
        # Missing implementations summary
        report.append("MISSING IMPLEMENTATIONS:")
        report.append("-" * 30)
        
        for section, data in self.analysis_report.items():
            if "missing" in data and data["missing"]:
                report.append(f"\n{section.upper()}:")
                for item in data["missing"]:
                    report.append(f"  MISSING: {item['file']} - {item['details']}")
        
        # Found implementations
        report.append("\nIMPLEMENTED COMPONENTS:")
        report.append("-" * 25)
        
        for section, data in self.analysis_report.items():
            if "found" in data and data["found"]:
                report.append(f"\n{section.upper()}:")
                for item in data["found"]:
                    report.append(f"  FOUND: {item['file']} - {item['details']}")
        
        # Detailed missing components
        missing_count = 0
        for section, data in self.analysis_report.items():
            if "missing" in data and data["missing"]:
                for item in data["missing"]:
                    missing_count += 1
                    report.append(f"  {item['file']}: {item['details']}")
        
        report.append(f"\nSUMMARY:")
        report.append(f"Total missing components: {missing_count}")
        
        return "\n".join(report)

def main():
    analyzer = CodebaseAnalyzer()
    analyzer.scan_codebase()
    print(analyzer.generate_report())

if __name__ == "__main__":
    main()