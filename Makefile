# Doom Index v2 — Makefile
# Common commands for development, training, and deployment

.PHONY: help install data train api dashboard demo test docker clean

PYTHON := python3
PIP := pip3
CONDA_ENV := doom

help:  ## Show this help message
	@echo "Doom Index v2 — Available Commands"
	@echo "===================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Done."

data:  ## Process Pushshift data with proper labels
	@echo "Processing Reddit data..."
	$(PYTHON) train_model_full_fixed.py --data_dir . --output data/processed_reddit_multimodal.csv
	@echo "Data processing complete."

train:  ## Train multimodal model (single GPU)
	@echo "Starting multimodal training..."
	$(PYTHON) train_multimodal.py \
		--data_path data/processed_reddit_multimodal.csv \
		--output_dir models/multimodal_doom \
		--epochs 10 \
		--batch_size 32 \
		--lr 2e-5

train-ddp:  ## Train with DDP (4 GPUs)
	@echo "Starting DDP training on 4 GPUs..."
	torchrun --nproc_per_node=4 train_multimodal.py \
		--data_path data/processed_reddit_multimodal.csv \
		--output_dir models/multimodal_doom \
		--epochs 15 \
		--batch_size 16 \
		--lr 2e-5 \
		--ddp \
		--fp16

train-hpc:  ## Submit HPC training job
	@echo "Submitting HPC job..."
	qsub hpc_multimodal_train.sh

api:  ## Start API server
	@echo "Starting API on http://localhost:8000"
	uvicorn api_v2:app --host 0.0.0.0 --port 8000 --reload

dashboard:  ## Start Streamlit dashboard
	@echo "Starting dashboard on http://localhost:8501"
	streamlit run dashboard/app.py

demo:  ## Run viva demo script
	@echo "Running demo..."
	$(PYTHON) demo.py

test:  ## Run unit tests
	@echo "Running tests..."
	pytest tests/test_multimodal.py -v

test-all:  ## Run all tests
	@echo "Running all tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing

docker-build:  ## Build Docker images
	@echo "Building Docker images..."
	docker-compose build

docker-up:  ## Start all services with Docker
	@echo "Starting services..."
	docker-compose up -d

docker-down:  ## Stop all Docker services
	@echo "Stopping services..."
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f api

migrate:  ## Run v1 -> v2 migration
	@echo "Running migration..."
	$(PYTHON) migrate.py --force

clean:  ## Clean generated files
	@echo "Cleaning..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov
	@echo "Clean complete."
