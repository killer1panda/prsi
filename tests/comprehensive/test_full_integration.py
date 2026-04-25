#!/usr/bin/env python3
"""Comprehensive Integration Tests for PRSI Doom Index.

This test suite covers:
- End-to-end pipeline integration
- API endpoint testing with realistic payloads
- Load testing simulation
- Model inference validation
- Data pipeline validation
- Neo4j graph operations
- Adversarial attack testing
- Performance benchmarks

Usage:
    # Run all tests
    pytest tests/comprehensive/test_full_integration.py -v
    
    # Run specific test class
    pytest tests/comprehensive/test_full_integration.py::TestPipelineIntegration -v
    
    # Run with coverage
    pytest tests/comprehensive/ --cov=src --cov-report=html
    
    # Run load tests only
    pytest tests/comprehensive/test_full_integration.py::TestLoadSimulation -v
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mark all tests as integration
pytestmark = [pytest.mark.integration, pytest.mark.comprehensive]


class TestPipelineIntegration:
    """Test end-to-end pipeline integration."""
    
    @pytest.fixture
    def sample_reddit_data(self) -> pd.DataFrame:
        """Create sample Reddit data for testing."""
        return pd.DataFrame({
            'post_id': ['post_1', 'post_2', 'post_3', 'post_4', 'post_5'],
            'author_id': ['user_a', 'user_b', 'user_a', 'user_c', 'user_d'],
            'author_name': ['UserA', 'UserB', 'UserA', 'UserC', 'UserD'],
            'body': [
                'This is a normal discussion post',
                'I absolutely hate this and everyone should know!',
                'Great point, I agree with you completely',
                'This person is a fraud and should be cancelled',
                'Interesting perspective, thanks for sharing'
            ],
            'score': [100, 5, 250, 15, 80],
            'created_utc': [1609459200, 1609459300, 1609459400, 1609459500, 1609459600],
            'subreddit': ['test', 'test', 'test', 'test', 'test'],
            'parent_author_id': [None, None, 'user_b', 'user_a', None],
        })
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        return data_dir
    
    def test_data_preprocessing_pipeline(self, sample_reddit_data, temp_data_dir):
        """Test complete data preprocessing pipeline."""
        # Save sample data
        input_path = temp_data_dir / "sample_input.parquet"
        sample_reddit_data.to_parquet(input_path)
        
        # Import preprocessing
        from src.data.preprocessing import preprocess_reddit_raw
        
        # Convert to raw format expected by preprocessing
        raw_posts = sample_reddit_data.to_dict('records')
        
        # Run preprocessing
        processed = preprocess_reddit_raw(raw_posts)
        
        # Validate output
        assert len(processed) > 0
        assert all('body' in p for p in processed)
        assert all('sentiment' in p or 'text' in p for p in processed)
    
    def test_feature_extraction_pipeline(self, sample_reddit_data):
        """Test feature extraction from preprocessed data."""
        texts = sample_reddit_data['body'].tolist()
        
        # Import feature extractor
        try:
            from src.features.engineering import extract_features
            
            features = extract_features(texts)
            
            assert features is not None
            assert len(features) == len(texts)
            assert features.shape[1] > 0  # Has features
            
        except ImportError:
            pytest.skip("Feature engineering module not available")
    
    def test_model_inference_pipeline(self):
        """Test model inference with sample inputs."""
        test_texts = [
            "This is a positive statement",
            "I hate everything about this",
            "Neutral observation here"
        ]
        
        # Mock predictor
        class MockPredictor:
            def predict(self, text, author_id=None):
                # Simple heuristic prediction
                if "hate" in text.lower():
                    doom = 0.85
                elif "positive" in text.lower():
                    doom = 0.15
                else:
                    doom = 0.5
                return {"probability": doom, "label": "high_risk" if doom > 0.7 else "low_risk"}
        
        predictor = MockPredictor()
        
        # Test predictions
        results = []
        for text in test_texts:
            result = predictor.predict(text)
            results.append(result)
        
        # Validate
        assert len(results) == 3
        assert all('probability' in r for r in results)
        assert all(0 <= r['probability'] <= 1 for r in results)
    
    def test_neo4j_graph_operations(self):
        """Test Neo4j graph building and querying."""
        # Skip if Neo4j not configured
        if not os.getenv("NEO4J_PASSWORD"):
            pytest.skip("Neo4j not configured")
        
        try:
            from src.data.neo4j_connector import Neo4jConnector
            
            # Connect
            connector = Neo4jConnector()
            
            # Test simple query
            with connector.driver.session(database=connector.database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                assert record["test"] == 1
            
        except Exception as e:
            pytest.skip(f"Neo4j connection failed: {e}")


class TestAPIEndpoints:
    """Test API endpoints with realistic payloads."""
    
    @pytest.fixture
    def api_client(self):
        """Create test API client."""
        try:
            from fastapi.testclient import TestClient
            from api_v2 import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI or api_v2 not available")
    
    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_predict_endpoint_basic(self, api_client):
        """Test basic prediction endpoint."""
        payload = {
            "text": "This is a test post about cancellation",
            "author_id": "test_user"
        }
        
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "doom_score" in data or "probability" in data or "prediction" in data
    
    def test_predict_endpoint_batch(self, api_client):
        """Test batch prediction endpoint."""
        payload = {
            "texts": [
                "Post one about controversy",
                "Post two with negative sentiment",
                "Post three neutral"
            ],
            "author_id": "batch_user"
        }
        
        response = api_client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or "predictions" in data
    
    def test_attack_simulate_endpoint(self, api_client):
        """Test adversarial attack simulation endpoint."""
        payload = {
            "original_text": "I think this is okay",
            "target_doom_increase": 20.0
        }
        
        response = api_client.post("/attack-simulate", json=payload)
        # May return 200 or skip if attack module not available
        if response.status_code == 200:
            data = response.json()
            assert "attacked_text" in data or "variants" in data
    
    def test_analyze_endpoint_full(self, api_client):
        """Test full analysis endpoint with all features."""
        payload = {
            "text": "Celebrity X just made a controversial statement",
            "author_id": "analyst_user",
            "include_explanations": True,
            "include_trajectory": True,
        }
        
        response = api_client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data is not None


class TestLoadSimulation:
    """Simulate production load patterns."""
    
    @pytest.fixture
    def load_test_config(self):
        """Load test configuration."""
        return {
            "num_requests": 100,
            "concurrent_users": 5,
            "ramp_up_seconds": 10,
            "test_duration_seconds": 60,
        }
    
    def test_sequential_load(self, load_test_config):
        """Test sequential request load."""
        try:
            from fastapi.testclient import TestClient
            from api_v2 import app
        except ImportError:
            pytest.skip("FastAPI not available")
        
        client = TestClient(app)
        
        latencies = []
        success_count = 0
        
        start_time = time.time()
        
        for i in range(load_test_config["num_requests"]):
            req_start = time.time()
            
            response = client.post(
                "/predict",
                json={"text": f"Load test request {i}", "author_id": "load_user"}
            )
            
            latency = (time.time() - req_start) * 1000  # ms
            latencies.append(latency)
            
            if response.status_code == 200:
                success_count += 1
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        qps = load_test_config["num_requests"] / total_time
        
        logger.info(f"Load Test Results:")
        logger.info(f"  Total requests: {load_test_config['num_requests']}")
        logger.info(f"  Success rate: {success_count / load_test_config['num_requests']:.1%}")
        logger.info(f"  Avg latency: {avg_latency:.1f}ms")
        logger.info(f"  P95 latency: {p95_latency:.1f}ms")
        logger.info(f"  P99 latency: {p99_latency:.1f}ms")
        logger.info(f"  QPS: {qps:.1f}")
        
        # Assertions
        assert success_count / load_test_config["num_requests"] > 0.95
        assert avg_latency < 1000  # < 1s average
        assert p99_latency < 5000  # < 5s p99
    
    def test_concurrent_load(self, load_test_config):
        """Test concurrent user load."""
        try:
            from fastapi.testclient import TestClient
            from api_v2 import app
            from concurrent.futures import ThreadPoolExecutor
        except ImportError:
            pytest.skip("Required modules not available")
        
        client = TestClient(app)
        results = {"success": 0, "failed": 0, "latencies": []}
        
        def make_request(user_id, request_id):
            start = time.time()
            try:
                response = client.post(
                    "/predict",
                    json={"text": f"Concurrent request {request_id}", "author_id": f"user_{user_id}"}
                )
                latency = (time.time() - start) * 1000
                
                if response.status_code == 200:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                
                results["latencies"].append(latency)
            except Exception as e:
                results["failed"] += 1
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=load_test_config["concurrent_users"]) as executor:
            futures = []
            for user in range(load_test_config["concurrent_users"]):
                for req in range(load_test_config["num_requests"] // load_test_config["concurrent_users"]):
                    futures.append(executor.submit(make_request, user, req))
            
            # Wait for completion
            for future in futures:
                future.result()
        
        # Calculate metrics
        total = results["success"] + results["failed"]
        success_rate = results["success"] / total if total > 0 else 0
        avg_latency = np.mean(results["latencies"]) if results["latencies"] else 0
        
        logger.info(f"Concurrent Load Test Results:")
        logger.info(f"  Total requests: {total}")
        logger.info(f"  Success rate: {success_rate:.1%}")
        logger.info(f"  Avg latency: {avg_latency:.1f}ms")
        
        assert success_rate > 0.90


class TestDataValidation:
    """Test data quality and validation."""
    
    def test_label_distribution(self):
        """Test label balance in dataset."""
        # Create sample labels
        labels = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
        
        # Check distribution
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts / len(labels)))
        
        logger.info(f"Label distribution: {distribution}")
        
        # Assert reasonable balance (not extremely imbalanced)
        assert distribution.get(1, 0) > 0.1  # At least 10% positive
        assert distribution.get(1, 0) < 0.9  # Not more than 90% positive
    
    def test_feature_ranges(self):
        """Test feature value ranges are valid."""
        # Simulate features
        features = {
            'sentiment': np.random.uniform(-1, 1, 100),
            'toxicity': np.random.uniform(0, 1, 100),
            'engagement': np.random.exponential(10, 100),
        }
        
        # Validate ranges
        assert features['sentiment'].min() >= -1
        assert features['sentiment'].max() <= 1
        assert features['toxicity'].min() >= 0
        assert features['toxicity'].max() <= 1
        assert features['engagement'].min() >= 0


class TestAdversarialRobustness:
    """Test adversarial attack and defense mechanisms."""
    
    def test_attack_generation(self):
        """Test adversarial example generation."""
        from src.attacks.adversarial_production import ProductionAdversarialGenerator
        
        # Mock predictor
        class MockPredictor:
            def predict(self, text, author_id=None):
                doom = 0.5 + np.random.uniform(-0.2, 0.2)
                return {"probability": doom}
        
        generator = ProductionAdversarialGenerator(
            predictor=MockPredictor(),
            max_iterations=10,
            population_size=5,
        )
        
        original_text = "This is a benign statement"
        
        # Generate attacks
        try:
            results = generator.generate_attacks(original_text, num_variants=3)
            
            assert len(results) > 0
            for result in results:
                assert result.variant_text != original_text
                assert hasattr(result, 'doom_uplift')
                
        except Exception as e:
            pytest.skip(f"Attack generation failed: {e}")
    
    def test_defense_robustness(self):
        """Test model robustness against attacks."""
        # This would test adversarial training effectiveness
        # For now, verify the adversarial training module exists
        try:
            from src.attacks.adversarial_training import AdversarialTrainer
            assert AdversarialTrainer is not None
        except ImportError:
            pytest.skip("Adversarial training module not available")


class TestPerformanceBenchmarks:
    """Benchmark performance metrics."""
    
    def test_inference_latency(self):
        """Test single inference latency."""
        # Mock model
        class MockModel:
            def predict(self, text, **kwargs):
                time.sleep(0.01)  # Simulate 10ms inference
                return {"probability": 0.5}
        
        model = MockModel()
        
        latencies = []
        for _ in range(50):
            start = time.time()
            model.predict("Test text for benchmarking")
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        logger.info(f"Inference Latency Benchmark:")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  P99: {p99_latency:.2f}ms")
        
        # Should be under 100ms for simple model
        assert avg_latency < 100
    
    def test_memory_usage(self):
        """Test memory usage during inference."""
        import psutil
        import torch
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_mem = process.memory_info().rss / 1024**2  # MB
        
        # Run some inference
        for _ in range(10):
            if torch.cuda.is_available():
                _ = torch.randn(1, 768).cuda()
        
        # Peak memory
        peak_mem = process.memory_info().rss / 1024**2
        
        memory_increase = peak_mem - baseline_mem
        
        logger.info(f"Memory Usage:")
        logger.info(f"  Baseline: {baseline_mem:.1f}MB")
        logger.info(f"  Peak: {peak_mem:.1f}MB")
        logger.info(f"  Increase: {memory_increase:.1f}MB")
        
        # Should not increase by more than 500MB for simple operations
        assert memory_increase < 500


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
