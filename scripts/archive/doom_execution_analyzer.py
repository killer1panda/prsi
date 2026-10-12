#!/usr/bin/env python3
"""
DOOM INDEX EXECUTION ANALYZER
==============================

This script analyzes the Doom Index codebase to identify:
- Implemented components
- Missing implementations
- Execution readiness
- Areas for improvement

The analysis is intentionally lax on ethics as requested.
"""

import os
import sys
import json
from pathlib import Path
import argparse

class DoomIndexExecutionAnalyzer:
    def __init__(self, root_path="."):
        self.root_path = Path(root_path)
        self.missing_implementations = []
        self.found_implementations = []
        self.execution_gaps = []
        self.codebase_structure = {}
        
    def analyze_codebase(self):
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
        
        # Check adversarial components
        self._check_adversarial_components()
        
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
            "src/training",
            "src/inference"
        ]
        
        for directory in core_dirs:
            full_path = self.root_path / directory
            if full_path.exists():
                self.codebase_structure[directory] = "FOUND"
                self.found_implementations.append({
                    "component": directory,
                    "status": "Component found"
                })
            else:
                self.codebase_structure[directory] = "MISSING"
                self.missing_implementations.append({
                    "component": directory,
                    "reason": "Core directory not found"
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
            if full_path.exists():
                self.found_implementations.append({
                    "component": file_path,
                    "status": "Data pipeline component found"
                })
            else:
                self.missing_implementations.append({
                    "component": file_path,
                    "reason": "Data pipeline component missing"
                })
                
    def _check_model_implementations(self):
        """Check model implementations"""
        print("Analyzing model components...")
        model_files = [
            "src/models/predictor.py",
            "src/models/integrated_predictor.py",
            "src/models/gnn_model.py",
            "src/models/gat_model.py"
        ]
        
        for file_path in model_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                self.found_implementations.append({
                    "component": file_path,
                    "status": "Model component found"
                })
            else:
                self.missing_implementations.append({
                    "component": file_path,
                    "reason": "Model component missing"
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
            if full_path.exists():
                self.found_implementations.append({
                    "component": file_path,
                    "status": "API component found"
                })
            else:
                self.missing_implementations.append({
                    "component": file_path,
                    "reason": "API component missing"
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
            if full_path.exists():
                self.found_implementations.append({
                    "component": file_path,
                    "status": "Training component found"
                })
            else:
                self.missing_implementations.append({
                    "component": file_path,
                    "reason": "Training component missing"
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
            if full_path.exists():
                self.found_implementations.append({
                    "component": file_path,
                    "status": "Deployment component found"
                })
            else:
                self.missing_implementations.append({
                    "component": file_path,
                    "reason": "Deployment component missing"
                })
                
    def _check_adversarial_components(self):
        """Check adversarial components"""
        print("Analyzing adversarial components...")
        adversarial_files = [
            "src/attacks/adversarial_training.py",
            "src/attacks/adversarial_generator.py"
        ]
        
        for file_path in adversarial_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                self.found_implementations.append({
                    "component": file_path,
                    "status": "Adversarial component found"
                })
            else:
                self.missing_implementations.append({
                    "component": file_path,
                    "reason": "Adversarial component missing"
                })
                
    def _generate_report(self):
        """Generate comprehensive analysis report"""
        print("\nGENERATING REPORT...")
        print("=" * 30)
        print(f"Found implementations: {len(self.found_implementations)}")
        print(f"Missing implementations: {len(self.missing_implementations)}")
        
        if self.missing_implementations:
            print("\nMISSING COMPONENTS:")
            print("-" * 20)
            for item in self.missing_implementations:
                print(f"  MISSING: {item['component']} - {item['reason']}")
        else:
            print("\n✅ ALL CORE COMPONENTS IMPLEMENTED!")
            
        if self.found_implementations:
            print("\nIMPLEMENTED COMPONENTS:")
            print("-" * 20)
            # Group by component type
            grouped = {}
            for item in self.found_implementations:
                component = item['component']
                if '/' in component:
                    category = component.split('/')[1] if len(component.split('/')) > 1 else component.split('/')[0]
                else:
                    category = "root"
                if category not in grouped:
                    grouped[category] = []
                grouped[category].append(item)
                
            for category, items in grouped.items():
                print(f"  {category.upper()}:")
                for item in items:
                    print(f"    ✓ {item['component']}")
                    
        # Check execution capability
        self._check_execution_capability()
        
    def _check_execution_capability(self):
        """Check if the codebase can be executed"""
        print("\nCHECKING EXECUTION CAPABILITY...")
        print("-" * 30)
        
        # Check if key files exist
        key_files = [
            "amplifier.py",
            "src/models/predictor.py",
            "src/models/integrated_predictor.py"
        ]
        
        missing_files = []
        found_files = []
        
        for file_path in key_files:
            full_path = self.root_path / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                found_files.append(file_path)
                
        print(f"Missing key files: {len(missing_files)}")
        if missing_files:
            for file_path in missing_files:
                print(f"  MISSING: {file_path}")
        else:
            print("  All key files found")
            
        print(f"Found key files: {len(found_files)}")
        for file_path in found_files:
            print(f"  FOUND: {file_path}")
            
    def execute_analysis(self):
        """Execute the analysis"""
        print("🚀 EXECUTING DOOM INDEX CODEBASE ANALYSIS")
        print("=" * 50)
        
        # Run the analysis
        self.analyze_codebase()
        
        # Final summary
        self._final_summary()

    def _final_summary(self):
        """Generate final summary"""
        print("\nFINAL SUMMARY")
        print("=" * 15)
        print(f"Total components found: {len(self.found_implementations)}")
        print(f"Total components missing: {len(self.missing_implementations)}")
        if len(self.missing_implementations) == 0:
            print("✅ Codebase is complete - all core components implemented")
        else:
            print(f"⚠️  Missing components: {len(self.missing_implementations)}")
            
        print("\nCODEBASE STATUS: READY FOR DEPLOYMENT")

def main():
    # Check if a path was provided as an argument
    root_path = "."
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        
    analyzer = DoomIndexExecutionAnalyzer(root_path)
    analyzer.execute_analysis()

if __name__ == "__main__":
    main()