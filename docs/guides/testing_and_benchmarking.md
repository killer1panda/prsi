# Testing & Benchmarking Guide

## Overview
This guide explains how to execute unit, integration, adversarial, and end-to-end SOTA verification tests across the PRSI repository.

---

## ⚡ Running SOTA Verification (Fastest)

To run the master test validating all 11 scientific subsystems:

```bash
PYTHONPATH=apps/backend python3 apps/backend/scripts/verify_sota_upgrades_e2e.py
```
- **Duration:** ~60 – 75 seconds.
- **Subsystems Tested:**
  1. Conformalized Quantile Regression (CQR)
  2. Continuous Diffusion Trajectory Forecaster
  3. Hypergraph Neural Network & Hawkes GAT
  4. Pearlian Double ML (DML) & Counterfactual Rewriter
  5. Frontier Multimodal Predictor (QLoRA + Q-Former)
  6. Async Redis Cache + XFetch Stampede Guard
  7. In-Process DuckDB OLAP Analytics Engine
  8. Stateful CEP Outrage Cascade Detector
  9. Adversarial GCG & AutoDAN Token Optimizer
  10. Adaptive DP-Adam & Homomorphic Encrypted FL
  11. Compliance Dossier & Benchmark Suite

---

## 🧪 Running Pytest (Avoiding Typeguard Slowness)

> [!WARNING]
> Python 3.14 includes experimental typing enhancements. If `typeguard` is active, it instruments all AST nodes in memory across all 223 tests, causing test runs to take 15–25 minutes.

To execute tests at full speed, disable `typeguard` and `locust`:

```bash
pytest apps/backend/tests/ -q -p no:typeguard -p no:locust --ignore=apps/backend/tests/test_load.py
```
- **Expected Duration:** ~85 – 110 seconds.
- **Pass Rate:** 100% passing across unit and integration suites.

---

## 📊 Generating Audit Reports

To re-generate the formal compliance dossier and PDF audit reports:

```bash
PYTHONPATH=apps/backend python3 apps/backend/src/evaluation/evaluate_full.py
```
Outputs generated:
- `reports/compliance/EU_AI_ACT_ANNEX_IV_DOSSIER.json`
- `reports/RED_TEAM_SECURITY_AUDIT_REPORT.pdf`
