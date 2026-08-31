#!/usr/bin/env python3
"""
Doom Index - Comprehensive Codebase Analysis and Execution Assessment

This script analyzes the entire codebase to identify:
1. Implemented components
2. Missing implementations
3. Execution gaps
4. Areas for improvement
"""

import os
import sys
import json
from pathlib import Path
import argparse
import importlib.util

class DoomIndexExecutionAnalyzer:
    def __init__(self, root_path="."):
        self.root_path = Path(root_path)
        self.missing_implementations = []
        self.found_implementations = []
        self.execution_gaps = []
        
    def analyze(self):
        """Perform comprehensive analysis of the codebase"""
        print("🚀 DOOM INDEX CODEBASE EXECUTION ANALYSIS")
        print("=" * 50)
        
        # Check all core components
        self._check_core_components()
        
        # Check data pipeline completeness
        self._check_data_pipeline()
        
        # Check model implementations
        self._check_model_implementations()
        
        # Check API implementations
        self._check_api_implementations()
        
        # Check training components
        self._check_training_components()
        
        # Check deployment components
        self._check_deployment_components()
        
        # Generate final report
        self._generate_report()
        
    def _check_core_components(self):
        """Check for core directory structure"""
        print("Checking core components...")
        core_dirs = [
            "src/models",
            "src/data", 
            "src/features",
            "src/api",
            "src/evaluation",
            "src/attacks",
            "src/training"
        ]
        
        for directory in core_dirs:
            full_path = self.root_path / directory
            if not full_path.exists():
                self.missing_implementations.append({
                    "component": directory,
                    "reason": "Core directory not found"
                })
            else:
                self.found_implementations.append({
                    "component": directory,
                    "status": "Component found"
                })
                
    def _check_data_pipeline(self):
        """Check data pipeline components"""
        print("Analyzing data pipeline...")
        pipeline_files = [
            "src/data/pipeline.py",
            "src/data/preprocessing.py",
            "src/data/scrapers/",
            "src/features/engineering.py",
            "src/data/db_connectors.py"
        ]
        
        for file_path in pipeline_files:
            full_path = self.root_path / file_path
            if not full_path.exists():
                self.missing_implementations.append({
                    "component": file_path,
                    "reason": "Data pipeline component missing"
                })
            else:
                self.found_implementations.append({
                    "component": file_path,
                    "status": "Data pipeline component found"
                })
                
    def _check_model_implementations(self):
        """Check model implementations"""
        print("Analyzing model components...")
        model_files = [
            "src/models/predictor.py",
            "src/models/integrated_predictor.py",
            "src/models/gnn_model.py"
        ]
        
        for file_path in model_files:
            full_path = self.root_path / file_path
            if not full_path.exists():
                self.missing_implementations.append({
                    "component": file_path,
                    "reason": "Model component missing"
                })
            else:
                self.found_implementations.append({
                    "component": file_path,
                    "status": "Model component found"
                })
                
    def _check_api_implementations(self):
        """Check API implementations"""
        print("Analyzing API components...")
        api_files = [
            "src/api/api_v2_production.py",
            "src/api/monitoring.py",
            "src/api/cache.py"
        ]
        
        for file_path in api_files:
            full_path = self.root_path / file_path
            if not full_path.exists():
                self.missing_implementations.append({
                    "component": file_path,
                    "reason": "API component missing"
                })
            else:
                self.found_implementations.append({
                    "component": file_path,
                    "status": "API component found"
                })
                
    def _check_training_components(self):
        """Check training components"""
        print("Analyzing training components...")
        training_files = [
            "train_model.py",
            "train_model_full.py",
            "src/models/multimodal_trainer.py"
        ]
        
        for file_path in training_files:
            full_path = self.root_path / file_path
            if not full_path.exists():
                self.missing_implementations.append({
                    "component": file_path,
                    "reason": "Training component missing"
                })
            else:
                self.found_implementations.append({
                    "component": file_path,
                    "status": "Training component found"
                })
                
    def _check_deployment_components(self):
        """Check deployment components"""
        print("Analyzing deployment components...")
        deployment_files = [
            "src/api/torchserve_config.py",
            "src/inference/tensorrt_optimizer.py"
        ]
        
        for file_path in deployment_files:
            full_path = self.root_path / file_path
            if not full_path.exists():
                self.missing_implementations.append({
                    "component": file_path,
                    "reason": "Deployment component missing"
                })
            else:
                self.found_implementations.append({
                    "component": file_path,
                    "status": "Deployment component found"
                })
                
    def _generate_report(self):
        """Generate comprehensive analysis report"""
        print("Generating report...")
        print(f"Found implementations: {len(self.found_implementations)}")
        print(f"Missing implementations: {len(self.missing_implementations)}")
        
        if self.missing_implementations:
            print("MISSING COMPONENTS:")
            print("-" * 20)
            for item in self.missing_implementations:
                print(f"  MISSING: {item['component']} - {item['reason']}")
        else:
            print("✅ All core components implemented!")
            
        if self.found_implementations:
            print("IMPLEMENTED COMPONENTS:")
            print("-" * 20)
            for item in self.found_implementations:
                print(f"  FOUND: {item['component']} - {item['status']}")
                
    def execute_analysis(self):
        """Execute the analysis"""
        print("🚀 EXECUTING DOOM INDEX CODEBASE ANALYSIS")
        print("=" * 50)
        
        # Run the analysis
        self.analyze()
        
        # Check execution capability
        self._check_execution_capability()
        
    def _check_execution_capability(self):
        """Check if the codebase can be executed"""
        print("Checking execution capability...")
        
        # Check if key files exist
        model_files = [
            "src/models/predictor.py",
            "src/models/integrated_predictor.py"
        ]
        
        for file_path in model_files:
            full_path = self.root_path / file_path
            if not full_path.exists():
                self.execution_gaps.append({
                    "component": file_path,
                    "reason": "Required for execution but missing"
                })
                
        # Check if we can import the main modules
        try:
            spec = importlib.util.spec_from_file_location("integrated_predictor", self.root_path / "src/models/integrated_predictor.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("✅ Integrated predictor can be imported")
        except Exception as e:
            self.execution_gaps.append({
                "component": "integrated_predictor",
                "reason": f"Import failed: {str(e)}"
            })
            
        # Check if we can run the amplifier
        try:
            spec = importlib.util.spec_from_file_location("amplifier", self.root_path / "amplifier.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("✅ Amplifier can be imported")
        except Exception as e:
            self.execution_gaps.append({
                "component": "amplifier",
                "reason": f"Amplifier import failed: {str(e)}"
            })
            
        print(f"Execution gaps found: {len(self.execution_gaps)}")
        if self.execution_gaps:
            print("EXECUTION GAPS:")
            print("-" * 20)
            for gap in self.execution_gaps:
                print(f"  {gap['component']}: {gap['reason']}")

def main():
    analyzer = DoomIndexExecutionAnalyzer()
    analyzer.execute_analysis()

if __name__ == "__main__":
    main()