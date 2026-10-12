# Getting Started & Local Setup

## Quickstart Guide

This guide walks you through setting up PRSI locally on macOS, Linux, or WSL for development and testing.

---

## 🛠️ Prerequisites
- **Python:** 3.10 to 3.14
- **Node.js:** v18+ (for Web/Desktop apps)
- **Git**

---

## 🚀 Installation

```bash
# 1. Clone repository
git clone https://github.com/killer1panda/prsi.git doom-index
cd doom-index

# 2. Install dependencies (CPU / Development mode)
pip install -r requirements.txt
```

> [!NOTE]
> By default, the environment runs with `HPC_MODE=0`. In this mode, heavy transformer checkpoints (~14GB Mistral, ~15GB Qwen) are bypassed with zero-weight local projection fallbacks so you can develop on storage-constrained laptops without downloading gigabytes of weights.

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root:

```bash
# Core Mode
HPC_MODE=0
DEVICE=cpu

# API & Security
API_KEY=doom_dev_key
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:8080

# Databases (Local Fallbacks will activate if services are offline)
MONGODB_URI=mongodb://localhost:27017/doom_index
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379/0
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

## 🧪 Running the Systems

### 1. Verify All SOTA Modules
Run the comprehensive 11-subsystem verification suite:

```bash
PYTHONPATH=apps/backend python3 apps/backend/scripts/verify_sota_upgrades_e2e.py
```
Expected output:
```text
🏁 MASTER SOTA VERIFICATION COMPLETE: 11/11 SUBSYSTEMS FULLY OPERATIONAL (100%)
```

### 2. Start FastAPI Production Server
```bash
python3 -m uvicorn apps.backend.src.api.api_v2_production:app --host 0.0.0.0 --port 8000 --reload
```
Access OpenAPI docs at: `http://localhost:8000/docs`

### 3. Launch Web Dashboard
```bash
cd apps/web
npm install
npm run dev
```
Open `http://localhost:3000` to interact with the real-time threat dashboard.
