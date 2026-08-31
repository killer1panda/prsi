# Predictive Social Doom Index + Shadowban Simulator

A machine learning system that predicts social media "cancellation" events (Doom Score 0-100) and simulates adversarial attacks that could push content toward being "canceled."

## Features

- **Doom Score Prediction**: Predict cancellation risk for social media users (0-100 scale)
- **Attack Simulator**: Generate adversarial content variants that maximize cancellation risk
- **Privacy-Preserving**: Differential privacy and federated learning support
- **Interactive Dashboard**: Visualize risk scores, network graphs, and attack simulations
- **Multimodal Analysis**: Text, network, and image feature fusion

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- 8GB+ RAM recommended
- API credentials (Twitter, Reddit)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/doom-index.git
cd doom-index

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API credentials

# Start databases (Docker)
docker-compose up -d
```

### Usage

```bash
# Run data collection
python -m src.data.pipeline

# Train model (Phase 3)
python -m src.models.training

# Start dashboard (Phase 6)
streamlit run src/dashboard/app.py
```

## Project Structure

```
doom-index/
├── config/              # Configuration files
│   └── config.yaml      # Main configuration
├── data/                # Data storage (gitignored)
│   ├── raw/             # Raw scraped data
│   ├── processed/       # Processed features
│   ├── models/          # Trained model files
│   └── exports/         # Data exports
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
│   ├── data/            # Data collection & processing
│   │   ├── scrapers/    # Platform scrapers
│   │   ├── db_connectors.py
│   │   ├── neo4j_connector.py
│   │   ├── pipeline.py
│   │   └── preprocessing.py
│   ├── features/        # Feature engineering
│   ├── models/          # ML models
│   ├── attacks/         # Attack simulator
│   ├── privacy/         # Privacy modules
│   ├── api/             # REST API
│   └── dashboard/       # Web dashboard
├── tests/               # Test suite
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── docker-compose.yml   # Docker services
├── Dockerfile           # Application container
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Sources   │     │  Feature Layer  │     │   Model Layer   │
│                 │     │                 │     │                 │
│ • Twitter/X     │────▶│ • Sentiment     │────▶│ • NLP Model     │
│ • Reddit        │     │ • Network       │     │ • GNN Model     │
│ • Instagram     │     │ • Time Series   │     │ • Fusion Model  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Dashboard    │     │  Privacy Layer  │     │ Attack Simulator│
│                 │     │                 │     │                 │
│ • Risk Heatmaps │◀────│ • Diff Privacy  │◀────│ • Text Attacks  │
│ • Network Graphs│     │ • Fed Learning  │     │ • Image Mutate  │
│ • Leaderboard   │     │                 │     │ • Evasion Opt   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## API Configuration

### Twitter/X API
1. Go to https://developer.twitter.com/en/portal/dashboard
2. Create a new project and app
3. Generate API keys and bearer token
4. Add credentials to `.env`

### Reddit API
1. Go to https://www.reddit.com/prefs/apps
2. Create a new app (script type)
3. Copy client_id and client_secret
4. Add credentials to `.env`

### Instagram
- Uses Instaloader for public profiles
- No API key required for basic usage
- Login optional for extended access

## Model Performance Targets

| Metric | Target |
|--------|--------|
| AUC-ROC | > 0.85 |
| F1 Score (High-Risk) | > 0.80 |
| Inference Latency | < 5 seconds |
| Calibration Error | < 0.10 |

## Development Roadmap

| Phase | Duration | Focus |
|-------|----------|-------|
| 1 | Weeks 1-4 | Foundation & Data Infrastructure |
| 2 | Weeks 5-8 | Feature Engineering |
| 3 | Weeks 9-14 | Model Development |
| 4 | Weeks 15-18 | Attack Simulator Module |
| 5 | Weeks 19-21 | Privacy & Optimization |
| 6 | Weeks 22-25 | Dashboard & Visualization |
| 7 | Weeks 26-28 | Testing & Documentation |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data/test_scrapers.py -v
```

## Docker Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild containers
docker-compose up -d --build
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This project is for educational and research purposes only. The attack simulator is designed to understand content moderation systems, not to circumvent them. Always respect platform terms of service and ethical guidelines.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{doom_index_2026,
  title = {Predictive Social Doom Index + Shadowban Simulator},
  author = {Project Team},
  year = {2026},
  url = {https://github.com/yourusername/doom-index}
}
```
