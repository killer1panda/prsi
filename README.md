# Doom Index - Predictive Social Doom Index

A machine learning system that predicts "cancellation events" on social media by analyzing Reddit posts for patterns of backlash, controversy, and public outrage.

## Features

- **Data Collection**: Automated scraping of Reddit posts with cancellation-related keywords
- **Sentiment Analysis**: VADER and RoBERTa-based sentiment scoring
- **Toxicity Detection**: Google Perspective API integration
- **ML Prediction**: RandomForest classifier for cancellation likelihood
- **REST API**: FastAPI endpoint for real-time analysis
- **Docker Deployment**: Containerized production-ready system

## Project Structure

```
doom-index/
├── src/
│   ├── data/           # Data collection and processing
│   ├── features/       # Feature engineering and analysis
│   ├── models/         # ML models and training
│   └── config.py       # Configuration management
├── api.py              # FastAPI application
├── docker-compose.yml  # Docker deployment
├── requirements.txt    # Python dependencies
├── models/             # Trained models (generated)
└── data/               # Dataset storage
```

## Installation

### Local Setup

```bash
# Clone and setup
cd doom-index
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Docker Setup

```bash
docker-compose up --build
```

## Usage

### Training the Model

```bash
# Process data and train model
python train_model_full.py

# Or use HPC for large-scale training
qsub hpc_train.sh
```

### Running the API

```bash
# Start API server
uvicorn api:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up
```

### API Endpoints

- `GET /` - Welcome message
- `POST /analyze` - Analyze text for cancellation risk
- `GET /health` - Health check

### Example API Usage

```python
import requests

response = requests.post("http://localhost:8000/analyze",
    json={"text": "This celebrity is facing massive backlash for their controversial statement"}
)

print(response.json())
# {
#   "prediction": 1,
#   "probability": 0.87,
#   "sentiment": {"compound": -0.65, ...},
#   "toxicity": {"toxicity": 0.72, ...}
# }
```

## Data Sources

- **Reddit Posts**: 2008-2012 submissions and comments (~1.1M total posts)
- **Filtered Dataset**: ~4,400 posts with cancellation keywords
- **Features**: Text analysis, sentiment scores, engagement metrics, temporal features

## Model Performance

- **Algorithm**: RandomForest Classifier
- **Accuracy**: ~84%
- **Features**: 15+ engineered features
- **Training Data**: 4,431 labeled posts

## HPC Usage

For large-scale processing, use the provided HPC scripts:

```bash
# Transfer to HPC
scp -r doom-index vivek.120542@10.16.1.50:/home/vivek.120542/

# On HPC
qsub hpc_train.sh
```

## Configuration

Set these environment variables in `.env`:

```bash
# Reddit API (optional)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret

# Perspective API
PERSPECTIVE_API_KEY=your_perspective_key

# Database (optional)
MONGODB_URI=mongodb://localhost:27017/doom_index
NEO4J_URI=bolt://localhost:7687
```

## Development

```bash
# Run tests
pytest

# Format code
black src/
isort src/

# Type checking
mypy src/
```

## License

This project is for educational and research purposes. Ensure compliance with platform terms of service and ethical guidelines.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If using this work for research, please cite:

```
Doom Index: Predictive Social Cancellation Analysis
[Your Name], 2026
```