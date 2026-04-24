# =============================================================================
# Doom Index Makefile
# Production-grade automation for all common operations
# =============================================================================

.PHONY: help install install-dev lint format test test-unit test-integration \
        test-load docker-build docker-up docker-down docker-logs \
        train train-hpc data-prepare graph-build evaluate export-onnx \
        clean docs viva-deploy migrate

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
RED := \033[31m
YELLOW := \033[33m
NC := \033[0m

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Help
# =============================================================================
help: ## Show this help message
	@echo "$(BLUE)Doom Index - Available Commands$(NC)"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	pip install -r requirements.txt

install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install -r requirements-dev.txt
	pre-commit install

install-hpc: ## Install HPC-specific dependencies (CUDA 12.x)
	@echo "$(BLUE)Installing HPC dependencies...$(NC)"
	pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
	pip install torch-geometric==2.4.0
	pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
	pip install transformers==4.36.0 datasets==2.15.0 accelerate==0.25.0
	pip install -r requirements.txt

# =============================================================================
# Code Quality
# =============================================================================
lint: ## Run all linters (flake8, mypy, bandit)
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports --show-error-codes
	bandit -r src/ -f json -o reports/bandit.json || true
	@echo "$(GREEN)Linting complete.$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/ scripts/
	isort src/ tests/ scripts/
	@echo "$(GREEN)Formatting complete.$(NC)"

format-check: ## Check formatting without modifying files
	@echo "$(BLUE)Checking formatting...$(NC)"
	black --check --diff src/ tests/ scripts/
	isort --check-only --diff src/ tests/ scripts/

# =============================================================================
# Testing
# =============================================================================
test: test-unit test-integration ## Run all tests

test-unit: ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-report=xml

test-integration: ## Run integration tests (requires services)
	@echo "$(BLUE)Running integration tests...$(NC)"
	docker-compose -f docker-compose-production.yml up -d redis neo4j mongodb
	pytest tests/test_integration.py -v --timeout=300
	docker-compose -f docker-compose-production.yml down

test-load: ## Run load tests with Locust
	@echo "$(BLUE)Running load tests...$(NC)"
	locust -f tests/test_load.py --host=http://localhost:8000 -u 100 -r 10 --run-time 5m --headless

test-security: ## Run security scans
	@echo "$(BLUE)Running security scans...$(NC)"
	bandit -r src/ -f json
	safety check -r requirements.txt

# =============================================================================
# Docker Operations
# =============================================================================
docker-build: ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose -f docker-compose-production.yml build

docker-up: ## Start all services with Docker Compose
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose -f docker-compose-production.yml up -d
	@echo "$(GREEN)Services started. API: http://localhost:8000 | Dashboard: http://localhost:8501 | Grafana: http://localhost:3000$(NC)"

docker-down: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	docker-compose -f docker-compose-production.yml down

docker-logs: ## Tail logs from all services
	docker-compose -f docker-compose-production.yml logs -f

docker-clean: ## Remove all containers, volumes, and images
	@echo "$(RED)Removing all Docker resources...$(NC)"
	docker-compose -f docker-compose-production.yml down -v --rmi all

# =============================================================================
# Data Pipeline
# =============================================================================
data-prepare: ## Run data preprocessing pipeline
	@echo "$(BLUE)Preparing data...$(NC)"
	python src/data/prepare.py --config configs/data.yaml
	python src/data/weak_labeling.py --input data/processed/features.csv --output data/processed/weak_labeled.csv

graph-build: ## Build Neo4j graph from processed data
	@echo "$(BLUE)Building Neo4j graph...$(NC)"
	python src/data/build_neo4j_graph_production.py \
		--data data/processed/weak_labeled.csv \
		--neo4j-uri bolt://localhost:7687 \
		--neo4j-user neo4j \
		--neo4j-password doomsday \
		--output-edges data/graph/edge_list.csv \
		--output-nodes data/graph/node_features.csv

# =============================================================================
# Training
# =============================================================================
train: ## Run full training pipeline locally
	@echo "$(BLUE)Running training pipeline...$(NC)"
	dvc repro

train-distilbert: ## Train DistilBERT text model
	@echo "$(BLUE)Training DistilBERT...$(NC)"
	python src/models/train_distilbert.py --config configs/distilbert.yaml

train-gnn: ## Train GraphSAGE/GNN model
	@echo "$(BLUE)Training GNN...$(NC)"
	python src/models/train_gnn.py --config configs/gnn.yaml

train-multimodal: ## Train multimodal fusion model
	@echo "$(BLUE)Training multimodal model...$(NC)"
	python src/models/train_multimodal.py --config configs/multimodal.yaml

train-adversarial: ## Run adversarial training
	@echo "$(BLUE)Running adversarial training...$(NC)"
	python src/attacks/adversarial_training.py --config configs/adv_train.yaml

train-hpc: ## Submit HPC training job
	@echo "$(BLUE)Submitting HPC training job...$(NC)"
	sbatch scripts/slurm_launcher.sh

train-hpc-status: ## Check HPC job status
	@squeue -u $(USER) | grep doom-index || echo "No active jobs"

# =============================================================================
# Evaluation & Export
# =============================================================================
evaluate: ## Run full evaluation suite
	@echo "$(BLUE)Running evaluation...$(NC)"
	python src/evaluation/evaluate_model.py --config configs/eval.yaml

export-onnx: ## Export trained model to ONNX
	@echo "$(BLUE)Exporting to ONNX...$(NC)"
	python scripts/export_onnx.py \
		--checkpoint models/robust/best_model.pt \
		--output models/doom_index.onnx \
		--quantize

# =============================================================================
# Viva/Demo Preparation
# =============================================================================
viva-deploy: docker-build docker-up ## Deploy full stack for viva demo
	@echo "$(GREEN)Viva deployment ready!$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Dashboard: http://localhost:8501"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Neo4j Browser: http://localhost:7474"

viva-demo: ## Run scripted viva demo
	@echo "$(BLUE)Running viva demo script...$(NC)"
	python scripts/viva_demo.py

viva-plots: ## Generate all viva presentation plots
	@echo "$(BLUE)Generating viva plots...$(NC)"
	python scripts/generate_viva_plots.py --output-dir viva_plots/

# =============================================================================
# Migration & Setup
# =============================================================================
migrate: ## Migrate all generated files to correct directories
	@echo "$(BLUE)Running migration...$(NC)"
	python scripts/migrate.py --force

validate: ## Validate setup (dependencies, models, connections)
	@echo "$(BLUE)Validating setup...$(NC)"
	python scripts/validate_setup.py

download-models: ## Download pretrained model artifacts
	@echo "$(BLUE)Downloading models...$(NC)"
	python scripts/download_models.py --cache-dir ./model_cache

# =============================================================================
# Monitoring
# =============================================================================
logs-api: ## Tail API logs
	docker-compose -f docker-compose-production.yml logs -f api

logs-dashboard: ## Tail dashboard logs
	docker-compose -f docker-compose-production.yml logs -f dashboard

metrics: ## Open Prometheus metrics
	@echo "$(GREEN)Prometheus: http://localhost:9090$(NC)"

grafana: ## Open Grafana dashboard
	@echo "$(GREEN)Grafana: http://localhost:3000$(NC)"

# =============================================================================
# Cleanup
# =============================================================================
clean: ## Clean generated files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/ .mypy_cache/ htmlcov/ dist/ build/
	rm -rf logs/*.log
	@echo "$(GREEN)Cleanup complete.$(NC)"

clean-all: clean docker-clean ## Deep clean everything
	@echo "$(RED)Deep clean complete.$(NC)"

# =============================================================================
# Documentation
# =============================================================================
docs: ## Generate documentation
	@echo "$(BLUE)Generating docs...$(NC)"
	cd docs && make html

# =============================================================================
# Git Helpers
# =============================================================================
git-tag-model: ## Tag current model version in Git
	@read -p "Enter version tag (e.g., v2.0.0): " version; \
	git tag -a $$version -m "Model release $$version"; \
	git push origin $$version
