# Doom Index v2.0 — Predictive Social Risk Assessment System

**A production-grade multimodal deep learning system for predicting social media backlash, cancellation events, and online outrage cascades.**

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1+-red)

---

## 🔥 Overview

The Doom Index is a comprehensive machine learning platform that analyzes social media posts using **multimodal deep learning** (NLP + Graph Neural Networks + Computer Vision) to predict the likelihood of cancellation events, backlash cascades, and viral outrage. 

Built for **H100 GPU clusters** with distributed training support, the system features adversarial robustness, differential privacy, federated learning simulation, real-time streaming inference, and production monitoring.

---

## ✨ Key Features

### 🧠 Multimodal Deep Learning
- **Text Encoder**: Fine-tuned DistilBERT with frozen layer optimization
- **Graph Neural Network**: GraphSAGE + GAT ensemble on Neo4j user interaction graphs
- **Vision Encoder**: CLIP ViT-B/32 for meme and image analysis
- **Temporal Modeling**: Temporal Graph Networks (TGN) for time-evolving social dynamics
- **Cross-Modal Fusion**: Attention-based fusion with gating mechanisms

### 🔒 Privacy & Security
- **Differential Privacy**: Opacus-based DP training with (ε, δ)-guarantees
- **Federated Learning**: Flower-based FL simulation for decentralized training
- **Adversarial Training**: TextAttack integration (TextFooler, BAE, PWWS) for robustness
- **Shadowban Attack Simulator**: Genetic algorithm-based adversarial example generation

### 🚀 Production Infrastructure
- **Real-Time Streaming**: Kafka + Apache Beam pipelines for live inference
- **Feature Store**: Redis (online) + Parquet (offline) with Feast integration
- **Model Serving**: ONNX Runtime + TensorRT optimization (<50ms p99 latency)
- **Distributed Training**: PyTorch DDP with mixed precision (FP16/BF16)
- **Experiment Tracking**: MLflow + Weights & Biases integration

### 📊 Monitoring & Evaluation
- **Drift Detection**: KS tests, PSI, and autoencoder reconstruction error
- **Fairness Auditing**: Disparate impact and equalized odds across demographics
- **Calibration**: Expected Calibration Error (ECE) monitoring
- **Dashboards**: Streamlit dashboard with 4 interactive tabs
- **Metrics Stack**: Prometheus + Grafana for system observability

### 🛡️ Advanced Toxicity Detection
- **ML-Based Ensemble**: Perspective API + HuggingFace Transformers + Rule-based
- **Category Breakdown**: Hate speech, threats, sexual harassment, profanity detection
- **Explainability**: Flagged token highlighting and confidence calibration
- **Rate Limiting**: Token bucket rate limiting (60 req/min default)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Doom Index v2.0 Architecture                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                      │
│  │  Text    │    │  Graph   │    │  Vision  │                      │
│  │DistilBERT│    │GraphSAGE │    │CLIP ViT  │                      │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘                      │
│       │               │               │                              │
│       └───────────────┼───────────────┘                              │
│                       ▼                                              │
│            ┌──────────────────────┐                                  │
│            │  Cross-Modal Fusion  │                                  │
│            │  + Attention Gating  │                                  │
│            └──────────┬───────────┘                                  │
│                       ▼                                              │
│            ┌──────────────────────┐                                  │
│            │  Temporal GNN (TGN)  │                                  │
│            └──────────┬───────────┘                                  │
│                       ▼                                              │
│            ┌──────────────────────┐                                  │
│            │  Classification Head │                                  │
│            └──────────┬───────────┘                                  │
│                       ▼                                              │
│            ┌──────────────────────┐                                  │
│            │  Doom Score (0-100)  │                                  │
│            └──────────────────────┘                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
doom-index/
├── src/
│   ├── api/                    # FastAPI v2 production server
│   │   ├── api_v2_production.py
│   │   ├── cache.py            # Redis caching layer
│   │   ├── monitoring.py       # Prometheus metrics
│   │   └── torchserve_config.py
│   ├── attacks/                # Adversarial attack simulation
│   │   ├── adversarial_generator.py
│   │   ├── adversarial_production.py
│   │   ├── adversarial_training.py
│   │   ├── textattack_full.py
│   │   └── toxicity_classifier.py
│   ├── benchmarks/             # H100 benchmarking
│   │   └── h100_benchmark.py
│   ├── dashboard/              # Streamlit dashboard
│   │   └── app_production.py
│   ├── data/                   # Data pipeline & scrapers
│   │   ├── scrapers/
│   │   ├── build_neo4j_graph_production.py
│   │   ├── neo4j_connector.py
│   │   ├── populate_neo4j_production.py
│   │   ├── preprocessing.py
│   │   ├── pushshift_ingestion.py
│   │   ├── unify_datasets.py
│   │   ├── weak_labeling.py
│   │   └── webdataset_converter.py
│   ├── evaluation/             # Comprehensive evaluation
│   │   ├── ab_testing.py
│   │   └── evaluate_full.py
│   ├── features/               # Feature engineering
│   │   ├── engineering.py
│   │   ├── feature_store.py
│   │   ├── graph_extractor.py
│   │   ├── sentiment.py
│   │   └── toxicity.py
│   ├── inference/              # Optimized inference
│   │   └── tensorrt_optimizer.py
│   ├── labels/                 # Label verification
│   │   └── llm_verifier.py
│   ├── models/                 # Core model architectures
│   │   ├── gnn_model.py
│   │   ├── gat_model.py
│   │   ├── temporal_gnn.py
│   │   ├── multimodal_trainer.py
│   │   ├── fusion.py
│   │   ├── vision_encoder.py
│   │   ├── multilingual.py
│   │   ├── interpretability.py
│   │   ├── fairness.py
│   │   ├── calibration.py
│   │   ├── drift_detector.py
│   │   ├── ensemble.py
│   │   ├── onnx_runtime.py
│   │   └── gnn_explainer.py
│   ├── privacy/                # Privacy-preserving ML
│   │   ├── dp_trainer.py       # Differential Privacy
│   │   └── fl_simulator.py     # Federated Learning
│   ├── registry/               # Model registry
│   │   └── model_registry.py
│   ├── streaming/              # Real-time pipelines
│   │   ├── kafka_pipeline.py
│   │   └── beam_pipeline.py
│   ├── tracking/               # Experiment tracking
│   │   └── experiment_tracker.py
│   ├── training/               # Training utilities
│   │   ├── deepspeed_config.py
│   │   ├── hyperparam_search_production.py
│   │   └── semi_supervised_trainer.py
│   └── validation/             # Data validation
│       └── data_validator.py
├── configs/                    # YAML configurations
│   ├── eval.yaml
│   ├── hpc_distilbert.yaml
│   └── multimodal.yaml
├── docker/                     # Docker configurations
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── Dockerfile.worker
├── monitoring/                 # Observability stack
│   ├── prometheus.yml
│   └── grafana/
├── tests/                      # Test suite
│   ├── unit/
│   ├── integration/
│   └── comprehensive/
├── docs/                       # Documentation
│   └── model_card.md
├── api_v2.py                   # FastAPI v2 entry point
├── train_multimodal.py         # Multimodal training script
├── evaluate_model.py           # Evaluation pipeline
├── demo.py                     # Interactive demo
├── docker-compose.yml          # Development compose
├── docker-compose-production.yml
├── Makefile                    # Command automation
├── dvc.yaml                    # DVC pipeline
└── requirements.txt            # Dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.x capable GPU (or CPU for testing)
- 16GB+ RAM (32GB+ recommended)
- 50GB+ disk space

### Installation

```bash
# Clone repository
git clone https://github.com/killer1panda/prsi.git
cd prsi

# Create conda environment
conda create -n doom python=3.10 -y
conda activate doom

# Install PyTorch with CUDA
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install remaining dependencies
pip install transformers==4.36.0 datasets==2.15.0 accelerate==0.25.0
pip install -r requirements.txt
```

### Docker Quick Start

```bash
# Start all services
docker-compose -f docker-compose-production.yml up -d

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# Grafana: http://localhost:3000
# Neo4j: http://localhost:7474
```

---

## 📖 Usage

### Training Pipeline

#### Single GPU Training

```bash
python train_multimodal.py \
    --data_path data/processed_reddit_multimodal.csv \
    --output_dir models/multimodal_doom \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5 \
    --fp16
```

#### Multi-GPU DDP Training (4x H100)

```bash
torchrun --nproc_per_node=4 train_multimodal.py \
    --data_path data/processed_reddit_multimodal.csv \
    --output_dir models/multimodal_doom \
    --epochs 15 \
    --batch_size 16 \
    --ddp \
    --fp16 \
    --grad_accum 4
```

#### HPC Cluster (PBS)

```bash
qsub hpc_multimodal_train.sh
```

### Full Pipeline with DVC

```bash
# Reproduce entire pipeline
dvc repro

# Run specific stages
dvc run -n prepare python src/data/prepare.py --config configs/data.yaml
dvc run -n extract_features python src/features/extract_all.py
dvc run -n train_multimodal python src/models/train_multimodal.py
```

### Running the API

```bash
# Start API server
python api_v2.py

# Or with uvicorn
uvicorn api_v2:app --host 0.0.0.0 --port 8000 --workers 4
```

### Launching Dashboard

```bash
streamlit run src/dashboard/app_production.py
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard redirect |
| `/health` | GET | Health check with system metrics |
| `/analyze` | POST | Multimodal doom score prediction |
| `/attack/simulate` | POST | Adversarial variant generation |
| `/leaderboard` | GET | Top doom scores leaderboard |
| `/predict/batch` | POST | Batch predictions (max 1000) |
| `/metrics` | GET | Prometheus metrics |

### Example API Request

```python
import requests

response = requests.post("http://localhost:8000/analyze",
    json={
        "text": "This celebrity is facing massive backlash for their controversial statement",
        "author_id": "user_12345",
        "followers": 50000,
        "verified": True
    }
)

print(response.json())
# {
#   "prediction": 1,
#   "probability": 0.91,
#   "doom_score": 87,
#   "risk_level": "CRITICAL",
#   "sentiment": {"compound": -0.72, "neg": 0.45, ...},
#   "toxicity": {"toxicity": 0.68, "categories": {...}},
#   "graph_embedding_norm": 2.34,
#   "text_embedding_norm": 15.67
# }
```

---

## 📊 Model Performance

### Benchmarks (Test Set)

| Metric | Baseline RF | Text-Only | Graph-Only | Multimodal v2 |
|--------|-------------|-----------|------------|---------------|
| **AUC-ROC** | 0.84 | 0.89 | 0.82 | **0.918** |
| **F1-Score (Macro)** | 0.72 | 0.81 | 0.75 | **0.874** |
| **F1-Score (High-Risk)** | 0.68 | 0.79 | 0.71 | **0.867** |
| **Precision@10** | 0.78 | 0.85 | 0.80 | **0.91** |
| **Calibration ECE** | 0.089 | 0.045 | 0.067 | **0.032** |

### Fairness Metrics

| Protected Attribute | Disparate Impact | Equalized Odds | Status |
|---------------------|------------------|----------------|--------|
| Language (en vs hi) | 0.87 | TPR diff: 0.04 | ⚠️ Monitor |
| User Type (verified) | 0.92 | TPR diff: 0.03 | ✅ Pass |
| Region (IN vs US) | 0.85 | TPR diff: 0.06 | ⚠️ Monitor |

---

## 📈 Data Pipeline

### Data Sources

- **Reddit**: Pushshift.io archives (2008-2026), ~1.1M posts filtered to ~450K cancellation-related
- **Twitter/X**: Public API samples, ~200K tweets with engagement metrics
- **Instagram**: Public profile metadata
- **Synthetic**: LLM-generated augmentation for rare events (~50K samples)

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 520,000 |
| Positive (Cancellation) | 130,000 |
| Negative | 390,000 |
| Avg Text Length | 145 tokens |
| Languages | en (78%), hi (12%), mixed (10%) |
| Time Span | 2018-01 to 2026-03 |

### Weak Labeling Pipeline

```python
from src.data.weak_labeling import WeakLabelGenerator

generator = WeakLabelGenerator(
    keyword_weights={"boycott": 0.8, "fired": 0.9, "cancelled": 0.85},
    engagement_threshold=100,
    sentiment_threshold=-0.5
)

labels = generator.generate_labels(df)
```

---

## 🛠️ Development

### Makefile Commands

```bash
make help              # Show all commands
make install           # Install production dependencies
make install-dev       # Install development dependencies
make lint              # Run linters (flake8, mypy, bandit)
make format            # Format code with black and isort
make test              # Run all tests
make test-unit         # Run unit tests
make test-integration  # Run integration tests
make test-load         # Run load tests with Locust
make docker-build      # Build Docker images
make docker-up         # Start all services
make train             # Run full training pipeline
make evaluate          # Run evaluation suite
make export-onnx       # Export model to ONNX
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v --cov=src --cov-report=html

# Integration tests (requires services)
pytest tests/integration/ -v --timeout=300

# Load tests
locust -f tests/test_load.py --host=http://localhost:8000 -u 100 -r 10

# Security scans
bandit -r src/ -f json
safety check -r requirements.txt
```

---

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
DOOM_API_URL=http://localhost:8000
DOOM_API_KEY=your-api-key
WORKERS=4
LOG_LEVEL=INFO

# Database Connections
MONGODB_URI=mongodb://admin:password@localhost:27017/doom_index
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
REDIS_URL=redis://localhost:6379/0

# Model Paths
MODEL_PATH=/app/models/doom_index.onnx
CONFIG_PATH=/app/models/model_config.pt

# External APIs (optional)
PERSPECTIVE_API_KEY=your-key
HF_TOKEN=your-huggingface-token

# Privacy Settings
DP_EPSILON=1.0
DP_DELTA=1e-5
```

### YAML Configuration Example

```yaml
# configs/multimodal.yaml
model:
  text_encoder: distilbert-base-uncased
  graph_hidden: 128
  fusion_type: cross_attention
  dropout: 0.3

training:
  learning_rate: 2e-5
  batch_size: 16
  epochs: 15
  warmup_steps: 500
  weight_decay: 0.01
  fp16: true
  grad_accum: 4

data:
  max_length: 512
  test_size: 0.1
  val_size: 0.1
  random_seed: 42
```

---

## 📦 Deployment

### Production Docker Compose

The production deployment includes:

- **API Server**: 3 replicas with GPU acceleration
- **Dashboard**: Streamlit frontend
- **Worker**: Celery background jobs (2 replicas)
- **Redis**: Cache + message broker
- **Neo4j**: Graph database with APOC + GDS plugins
- **MongoDB**: Document store
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Nginx**: Reverse proxy + load balancer

```bash
docker-compose -f docker-compose-production.yml up -d
```

### Kubernetes Deployment (Optional)

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/services.yaml
```

---

## 📝 Documentation

- **[Model Card](docs/model_card.md)**: Detailed model information, intended use, and ethical considerations
- **[Quick Start Guide](QUICKSTART.md)**: 30-minute setup guide
- **[Integration Guide](INTEGRATION_GUIDE.md)**: Step-by-step integration instructions
- **[HPC Execution Guide](HPC_EXECUTION_GUIDE.md)**: H100 cluster usage guide
- **[Data Pipeline](README_DATA_PIPELINE.md)**: Data processing documentation

---

## ⚖️ Ethical Considerations

### Potential Risks

1. **Surveillance**: Could be misused for mass monitoring of dissent
2. **Bias**: May overpredict risk for non-English speakers and marginalized communities
3. **Chilling Effect**: Users may self-censor if aware of risk scoring
4. **Adversarial Gaming**: Bad actors could optimize harmful content

### Mitigations

- ✅ **Differential Privacy**: Training with ε=1.0 noise
- ✅ **Federated Learning**: Decentralized training simulation
- ✅ **Transparency**: Open-source release with model cards
- ✅ **Human-in-the-Loop**: High-risk predictions flagged for review
- ✅ **Rate Limiting**: API throttling to prevent bulk surveillance
- ✅ **Fairness Auditing**: Regular bias assessments

---

## 📄 License

MIT License — Research Use Only

This project is for **educational and research purposes**. Ensure compliance with:
- Platform terms of service (Reddit, Twitter, etc.)
- Data protection regulations (GDPR, CCPA)
- Ethical AI guidelines
- Institutional review board (IRB) approval for human subjects research

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`make test`)
5. Run linters (`make lint`)
6. Update documentation
7. Submit a pull request

---

## 📚 Citation

If using this work in your research, please cite:

```bibtex
@software{doom_index_2026,
  title = {Doom Index: Predictive Social Risk Assessment},
  author = {Doom Index Team},
  year = {2026},
  url = {https://github.com/killer1panda/prsi},
  version = {2.0.0}
}
```

---

## 🎯 Roadmap

### Completed (v2.0)
- ✅ Multimodal fusion (text + graph + vision)
- ✅ Temporal Graph Networks (TGN)
- ✅ Adversarial training for robustness
- ✅ Differential privacy with Opacus
- ✅ Federated learning simulation
- ✅ Real-time streaming (Kafka + Beam)
- ✅ Fairness auditing pipeline
- ✅ Drift detection system
- ✅ Production monitoring (Prometheus + Grafana)

### Planned (v2.1+)
- 🔄 Cross-attention fusion improvements
- 🔄 Multilingual support expansion (10+ languages)
- 🔄 SHAP interpretability integration
- 🔄 Active learning pipeline
- 🔄 Reinforcement learning from human feedback (RLHF)
- 🔄 Edge deployment (TensorFlow Lite, CoreML)

---

## 👥 Team & Acknowledgments

Developed by the Predictive Social Risk Research Group.

Special thanks to:
- PyTorch Geometric team for GNN tools
- Hugging Face for transformer models
- Neo4j team for graph database support
- Open source contributors

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/killer1panda/prsi/issues)
- **Discussions**: [GitHub Discussions](https://github.com/killer1panda/prsi/discussions)
- **Email**: doom-index-team@university.edu

---

**Built with ❤️ for responsible AI research**

*Last Updated: May 2026*