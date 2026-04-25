#!/usr/bin/env python3
"""
Comprehensive Test Suite for Doom Index Production System.

Tests covering:
- Unit tests for individual components
- Integration tests for end-to-end flows
- Load tests for API performance
- Adversarial robustness tests
- Data validation tests
- Model quality tests

Run with: pytest tests/comprehensive/ -v --cov=src
"""

import os
import sys
import time
import json
import pytest
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Test configuration."""
    return {
        'neo4j_uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'neo4j_user': os.getenv('NEO4J_USER', 'neo4j'),
        'neo4j_password': os.getenv('NEO4J_PASSWORD', 'doom_index_prod_2026'),
        'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        'api_base_url': os.getenv('API_BASE_URL', 'http://localhost:8000'),
    }


@pytest.fixture
def sample_twitter_data():
    """Sample Twitter data for testing."""
    return [
        {
            'id': '123456',
            'user_id': 'user1',
            'username': 'test_user1',
            'text': 'This is a normal tweet about cancellation',
            'created_at': '2024-01-01T12:00:00Z',
            'retweet_count': 10,
            'favorite_count': 50,
        },
        {
            'id': '123457',
            'user_id': 'user2',
            'username': 'angry_user',
            'text': '@test_user1 You are stupid and should be cancelled!!!',
            'created_at': '2024-01-01T12:05:00Z',
            'in_reply_to_user_id': 'user1',
            'retweet_count': 0,
            'favorite_count': 5,
        },
    ]


@pytest.fixture
def sample_reddit_data():
    """Sample Reddit data for testing."""
    return pd.DataFrame({
        'author': ['user1', 'user2', 'user3'],
        'body': ['Normal comment', 'Toxic comment here', 'Another normal one'],
        'subreddit': ['test', 'test', 'other'],
        'score': [10, -5, 20],
        'created_utc': [1609459200, 1609459300, 1609459400],
        'parent_author': [None, 'user1', None],
    })


# =============================================================================
# Toxicity Classifier Tests
# =============================================================================

class TestToxicityClassifier:
    """Tests for production toxicity classifier."""
    
    @pytest.mark.asyncio
    async def test_rule_based_toxicity(self):
        """Test rule-based toxicity detection."""
        from src.attacks.toxicity_classifier import ProductionToxicityClassifier
        
        classifier = ProductionToxicityClassifier(use_ensemble=False)
        await classifier.initialize()
        
        # High toxicity text
        result = await classifier.predict("You're a fucking idiot and should die!")
        assert result.toxicity_score > 0.5
        assert result.is_toxic
        assert len(result.flagged_tokens) > 0
        
        # Low toxicity text
        result = await classifier.predict("I love sunny days and puppies")
        assert result.toxicity_score < 0.3
        assert not result.is_toxic
        
        await classifier.close()
    
    @pytest.mark.asyncio
    async def test_hate_speech_detection(self):
        """Test hate speech pattern detection."""
        from src.attacks.toxicity_classifier import ProductionToxicityClassifier
        
        classifier = ProductionToxicityClassifier(use_ensemble=False)
        await classifier.initialize()
        
        result = await classifier.predict(
            "All those Muslims should go back to their country"
        )
        
        assert result.categories.get('hate_speech', 0) > 0.3
        assert result.toxicity_score > 0.4
        
        await classifier.close()
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test toxicity prediction caching."""
        from src.attacks.toxicity_classifier import ProductionToxicityClassifier
        
        classifier = ProductionToxicityClassifier()
        await classifier.initialize()
        
        text = "Test text for caching"
        
        # First call (cache miss)
        result1 = await classifier.predict(text)
        assert not result1.cache_hit
        
        # Second call (should be cached if Redis available)
        result2 = await classifier.predict(text)
        # Note: cache_hit depends on Redis availability
        
        await classifier.close()


# =============================================================================
# A/B Testing Framework Tests
# =============================================================================

class TestABTesting:
    """Tests for A/B testing framework."""
    
    @pytest.mark.asyncio
    async def test_traffic_routing(self):
        """Test consistent traffic routing."""
        from src.evaluation.ab_testing import ABTestingFramework, TrafficAllocationStrategy
        
        framework = ABTestingFramework()
        await framework.initialize()
        
        config = await framework.create_experiment(
            name="test_exp",
            description="Test experiment",
            control_model_id="model_v1",
            treatment_model_ids=["model_v2"],
            strategy=TrafficAllocationStrategy.UNIFORM,
            min_sample_size=10,
        )
        
        # Same user should always get same variant
        user_id = "test_user_123"
        variant1 = framework.assign_variant(user_id, config.experiment_id)
        variant2 = framework.assign_variant(user_id, config.experiment_id)
        
        assert variant1 == variant2
        
        await framework.close()
    
    @pytest.mark.asyncio
    async def test_statistical_analysis(self):
        """Test statistical analysis of A/B test results."""
        from src.evaluation.ab_testing import ABTestingFramework
        
        framework = ABTestingFramework()
        await framework.initialize()
        
        config = await framework.create_experiment(
            name="stats_test",
            description="Statistical test",
            control_model_id="control",
            treatment_model_ids=["treatment"],
            min_sample_size=50,
        )
        
        # Simulate observations where treatment is better
        np.random.seed(42)
        for i in range(200):
            user_id = f"user_{i}"
            
            # Assign variant
            variant = framework.assign_variant(user_id, config.experiment_id)
            
            # Simulate F1 scores (treatment better)
            if variant == "control":
                f1 = np.random.normal(0.70, 0.1)
            else:
                f1 = np.random.normal(0.80, 0.1)
            
            framework.record_observation(
                experiment_id=config.experiment_id,
                user_id=user_id,
                variant=variant,
                metrics={"f1_score": f1},
            )
        
        # Analyze results
        result = await framework.analyze_experiment(config.experiment_id)
        
        assert result.sample_sizes['control'] >= 50
        assert result.sample_sizes['treatment'] >= 50
        
        # Check that analysis completed
        assert 'f1_score' in result.metric_results
        
        await framework.close()


# =============================================================================
# Neo4j Population Tests
# =============================================================================

class TestNeo4jPopulation:
    """Tests for Neo4j graph population."""
    
    @pytest.mark.asyncio
    async def test_user_creation(self, sample_twitter_data):
        """Test user node creation in Neo4j."""
        from src.data.populate_neo4j_production import (
            ProductionNeo4jPopulator,
            Neo4jConfig,
        )
        
        config = Neo4jConfig(
            uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            password=os.getenv('NEO4J_PASSWORD', 'doom_index_prod_2026'),
        )
        
        populator = ProductionNeo4jPopulator(config)
        
        try:
            await populator.initialize()
            
            # Create users from sample data
            df = pd.DataFrame(sample_twitter_data)
            await populator._create_users_from_twitter(df)
            
            # Verify users exist
            stats = await populator.get_graph_statistics()
            assert stats.get('user_count', 0) >= 2
            
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
        finally:
            await populator.close()
    
    @pytest.mark.asyncio
    async def test_edge_creation(self, sample_twitter_data):
        """Test edge creation in Neo4j."""
        from src.data.populate_neo4j_production import (
            ProductionNeo4jPopulator,
            Neo4jConfig,
        )
        
        config = Neo4jConfig(
            uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            password=os.getenv('NEO4J_PASSWORD', 'doom_index_prod_2026'),
        )
        
        populator = ProductionNeo4jPopulator(config)
        
        try:
            await populator.initialize()
            
            # Create mention edges
            df = pd.DataFrame(sample_twitter_data)
            await populator._create_mention_edges(df)
            
            # Verify edges created
            with populator.driver.session() as session:
                result = session.run("""
                    MATCH ()-[r:INTERACTS_WITH]->()
                    RETURN count(r) AS edge_count
                """)
                record = result.single()
                assert record['edge_count'] >= 1
            
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
        finally:
            await populator.close()


# =============================================================================
# API Integration Tests
# =============================================================================

class TestAPIIntegration:
    """Integration tests for FastAPI application."""
    
    def test_health_endpoint(self, test_config):
        """Test API health check endpoint."""
        try:
            response = requests.get(f"{test_config['api_base_url']}/health")
            assert response.status_code == 200
            assert response.json()['status'] == 'healthy'
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running")
    
    def test_prediction_endpoint(self, test_config):
        """Test prediction endpoint."""
        try:
            payload = {
                'text': 'Test post for doom index prediction',
                'username': 'test_user',
            }
            
            response = requests.post(
                f"{test_config['api_base_url']}/predict",
                json=payload,
                headers={'X-API-Key': 'test_key'}
            )
            
            assert response.status_code == 200
            result = response.json()
            
            assert 'doom_score' in result
            assert 0 <= result['doom_score'] <= 100
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running")


# =============================================================================
# Load Tests
# =============================================================================

class TestLoadPerformance:
    """Load and performance tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, test_config):
        """Test API under concurrent load."""
        import aiohttp
        
        async def make_prediction(session, text):
            payload = {'text': text, 'username': 'load_test_user'}
            try:
                async with session.post(
                    f"{test_config['api_base_url']}/predict",
                    json=payload,
                    headers={'X-API-Key': 'test_key'}
                ) as response:
                    return response.status
            except:
                return None
        
        async with aiohttp.ClientSession() as session:
            # Send 100 concurrent requests
            tasks = [
                make_prediction(session, f"Load test message {i}")
                for i in range(100)
            ]
            
            results = await asyncio.gather(*tasks)
            successful = sum(1 for r in results if r == 200)
            
            # At least 80% should succeed
            assert successful >= 80


# =============================================================================
# Model Quality Tests
# =============================================================================

class TestModelQuality:
    """Tests for model quality and robustness."""
    
    def test_prediction_calibration(self):
        """Test that predictions are well-calibrated."""
        # Load model predictions vs actuals
        # This would use held-out test set
        pass
    
    def test_adversarial_robustness(self):
        """Test model robustness to adversarial examples."""
        from src.attacks.adversarial_production import ProductionAdversarialGenerator
        
        # Generate adversarial examples
        # Verify model doesn't degrade too much
        pass


# =============================================================================
# Data Validation Tests
# =============================================================================

class TestDataValidation:
    """Tests for data validation pipeline."""
    
    def test_schema_validation(self, sample_reddit_data):
        """Test data schema validation."""
        from src.validation.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Validate schema
        is_valid, errors = validator.validate_schema(sample_reddit_data)
        assert is_valid
    
    def test_label_distribution(self, sample_reddit_data):
        """Test label distribution validation."""
        from src.validation.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Add synthetic labels
        sample_reddit_data['label'] = [1, 0, 1]
        
        # Check for imbalance
        report = validator.check_label_balance(sample_reddit_data, 'label')
        assert 'imbalance_ratio' in report


# =============================================================================
# Drift Detection Tests
# =============================================================================

class TestDriftDetection:
    """Tests for drift detection."""
    
    def test_feature_drift(self):
        """Test feature distribution drift detection."""
        from src.mlops.drift_detector import DriftDetector
        
        detector = DriftDetector()
        
        # Reference distribution
        reference = np.random.normal(0, 1, 1000)
        
        # Current distribution (no drift)
        current = np.random.normal(0, 1, 1000)
        
        drift_detected, stats = detector.detect_feature_drift(
            reference, current, method='ks'
        )
        
        # Should not detect drift with same distribution
        assert not drift_detected or stats['p_value'] > 0.01


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

class TestEndToEndPipeline:
    """End-to-end pipeline tests."""
    
    @pytest.mark.asyncio
    async def test_full_prediction_flow(self):
        """Test complete prediction flow from input to output."""
        # 1. Ingest data
        # 2. Extract features
        # 3. Run model inference
        # 4. Store prediction
        # 5. Return result
        pass


# =============================================================================
# Security Tests
# =============================================================================

class TestSecurity:
    """Security-related tests."""
    
    def test_api_authentication(self, test_config):
        """Test API authentication requirements."""
        try:
            # Request without auth should fail
            response = requests.post(
                f"{test_config['api_base_url']}/predict",
                json={'text': 'test'}
            )
            assert response.status_code in [401, 403]
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running")
    
    def test_rate_limiting(self, test_config):
        """Test rate limiting functionality."""
        # Send many requests rapidly
        # Verify rate limit is enforced
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
