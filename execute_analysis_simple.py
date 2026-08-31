#!/usr/bin/env python3
"""
Doom Index - Execution and Gap Analysis
"""

# Core modules
import os
import sys
import json
from pathlib import Path
import torch
import importlib.util

class DoomIndexAnalyzer:
    def __init__(self, root_path="."):
        self.root_path = Path(root_path)
        self.missing_implementations = []
        self.found_implementations = []
        self.execution_gaps = []
        self.codebase_structure = {}
        
    def scan_codebase(self):
        """Scan the entire codebase structure"""
        print("Scanning codebase structure...")
        
        # Core directories
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
            else:
                self.codebase_structure[directory] = "MISSING"
                self.missing_implementations.append({
                    "component": directory,
                    "reason": "Core directory not found"
                })
                
    def check_execution_readiness(self):
        """Check if the codebase is ready for execution"""
        print("Checking execution readiness...")
        
        # Check if all required files exist
        required_files = [
            "amplifier.py",
            "src/models/predictor.py",
            "src/models/integrated_predictor.py",
            "src/models/gnn_model.py"
        ]
        
        for file_path in required_files:
            full_path = self.root_path / file_path
            if not full_path.exists():
                self.execution_gaps.append({
                    "component": file_path,
                    "reason": "Required for execution but missing"
                })
                
        # Check if we can import the main modules
        try:
            # Try importing amplifier
            spec = importlib.util.spec_from_file_location("amplifier", self.root_path / "amplifier.py")
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("✅ Amplifier can be imported")
        except Exception as e:
            self.execution_gaps.append({
                "component": "amplifier",
                "reason": f"Import failed: {str(e)}"
            })
            
        # Check if we can import predictor
        try:
            spec = importlib.util.spec_from_file_location("predictor", self.root_path / "src/models/predictor.py")
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("✅ Predictor can be imported")
        except Exception as e:
            self.execution_gaps.append({
                "component": "predictor",
                "reason": f"Import failed: {str(e)}"
            })
            
        # Check if we can import integrated predictor
        try:
            spec = importlib.util.spec_from_file_location("integrated_predictor", self.root_path / "src/models/integrated_predictor.py")
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("✅ Integrated predictor can be imported")
        except Exception as e:
            self.execution_gaps.append({
                "component": "integrated_predictor",
                "reason": f"Import failed: {str(e)}"
            })
            
        print(f"Execution gaps found: {len(self.execution_gaps)}")
        if self.execution_gaps:
            print("EXECUTION Gaps:")
            for gap in self.execution_gaps:
                print(f"  {gap['component']}: {gap['reason']}")
                
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\nDOOM INDEX CODEBASE ANALYSIS REPORT")
        print("=" * 40)
        
        print("CODEBASE STRUCTURE:")
        print("-" * 20)
        for component, status in self.codebase_structure.items():
            print(f"  {component}: {status}")
            
        print(f"\nMISSING IMPLEMENTATIONS: {len(self.missing_implementations)} items")
        if self.missing_implementations:
            for item in self.missing_implementations:
                print(f"  MISSING: {item['component']} - {item['reason']}")
        else:
            print("  All core components implemented!")
            
        print(f"\nEXECUTION GAPS: {len(self.execution_gaps)} items")
        if self.execution_gaps:
            for gap in self.execution_gaps:
                print(f"  {gap['component']}: {gap['reason']}")
        else:
            print("  ✅ All components ready for execution")
            
    def run_analysis(self):
        """Run the complete analysis"""
        print("🚀 DOOM INDEX EXECUTION ANALYSIS")
        print("=" * 40)
        
        # Scan codebase
        self.scan_codebase()
        
        # Check execution
        self.check_execution_readiness()
        
        # Generate report
        self.generate_report()

# Create the analyzer and run
analyzer = DoomIndexAnalyzer()
analyzer.run_analysis()