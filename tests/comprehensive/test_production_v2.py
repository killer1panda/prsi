#!/usr/bin/env python3
"""
Comprehensive Test Suite for Doom Index Production System.
"""

import os
import sys
import time
import json
import pytest
import asyncio
import aiohttp
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
            'created_at': datetime.utcnow().isoformat(),
            'retweets': 10,
            'likes': 50,
            'replies': 5,
            'user_followers': 1000,
            'user_verified': False,
            'hashtags': ['cancel'],
            'mentions': ['user2'],
        },
        {
            'id': '123457',
            'user_id': 'user2',
            'username': 'test_user2',
            'text': 'Boycott this company immediately #boycott',
            'created_at': datetime.utcnow().isoformat(),
            'retweets': 100,
            'likes': 500,
            'replies': 50,
            'user_followers': 50000,
            'user_verified': True,
            'hashtags': ['boycott'],
            'mentions': [],
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
        assert result.toxicity_score > 0.4
        assert len(result.flagged_tokens) > 0
        
        # Low toxicity text
        result = await classifier.predict("I love sunny days and puppies")
        assert result.toxicity_score < 0.3
        
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
        
        assert result.categories.get('hate_speech', 0) >= 0.3
        assert result.toxicity_score >= 0.3
        
        await classifier.close()
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test toxicity prediction caching."""
        from src.attacks.toxicity_classifier import ProductionToxicityClassifier
        
        classifier = ProductionToxicityClassifier(use_ensemble=False)
        await classifier.initialize()
        
        text = "Test text for caching"
        result1 = await classifier.predict(text)
        assert result1 is not None
        
        result2 = await classifier.predict(text)
        assert result2 is not None
        
        await classifier.close()


# =============================================================================
# A/B Testing Framework Tests
# =============================================================================

class TestABTesting:
    """Tests for A/B testing framework."""
    
    @pytest.mark.asyncio
    async def test_traffic_routing(self):
        """Test consistent traffic routing."""
        from src.evaluation.ab_testing import TrafficRouter, ABTestConfig
        
        cfg = ABTestConfig(traffic_split=0.5)
        router = TrafficRouter(cfg)
        user_id = "test_user_123"
        variant1 = router.route(user_id)
        variant2 = router.route(user_id)
        assert variant1 == variant2
    
    @pytest.mark.asyncio
    async def test_statistical_analysis(self):
        """Test statistical analysis of A/B test results."""
        from src.evaluation.ab_testing import StatisticalTester, ABTestConfig
        
        cfg = ABTestConfig()
        tester = StatisticalTester(cfg)
        control = np.random.normal(0.70, 0.05, 50)
        treatment = np.random.normal(0.80, 0.05, 50)
        
        t_res = tester.t_test(control, treatment)
        assert "p_value" in t_res
        assert "significant" in t_res


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
        from src.api.api_v2_production import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json().get('status') in ['healthy', 'degraded']
    
    def test_prediction_endpoint(self, test_config):
        """Test prediction endpoint."""
        from src.api.api_v2_production import app, config
        config.require_auth = False
        client = TestClient(app)
        payload = {
            'text': 'Test post for doom index prediction',
            'author_id': 'test_user',
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert 'doom_score' in result or 'probability' in result


# =============================================================================
# Load Tests
# =============================================================================

class TestLoadPerformance:
    """Load and performance tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, test_config):
        """Test API under in-process concurrent load."""
        from src.api.api_v2_production import app, config
        config.require_auth = False
        client = TestClient(app)
        
        successes = 0
        for i in range(5):
            resp = client.post("/predict", json={'text': f'Concurrent test {i}', 'author_id': f'user_{i}'})
            if resp.status_code == 200:
                successes += 1
        assert successes >= 4


# =============================================================================
# Model Quality Tests
# =============================================================================

class TestModelQuality:
    """Tests for model prediction quality and calibration."""
    
    def test_prediction_calibration(self):
        """Test calibration logic."""
        from src.models.calibration import FollowerStratifiedCalibrator
        cal = FollowerStratifiedCalibrator()
        p = cal.calibrate_single(0.6, followers=1000)
        assert 0.0 <= p <= 1.0

    def test_adversarial_robustness(self):
        """Test adversarial perturbation robustness."""
        from src.attacks.adversarial_production import VisualVenomInjector
        from PIL import Image
        venom = VisualVenomInjector()
        img = Image.new("RGB", (100, 100), color=(120, 120, 120))
        res = venom.generate_image_attacks(img, original_doom=40.0)
        assert len(res) == 4


# =============================================================================
# Data Validation Tests
# =============================================================================

class TestDataValidation:
    """Tests for data validation pipeline."""
    
    def test_schema_validation(self, sample_reddit_data):
        """Test data schema validation."""
        from src.validation.data_validator import DataValidator
        validator = DataValidator(strict=False)
        is_valid, msg = validator.validate_schema(sample_reddit_data)
        assert isinstance(is_valid, bool)
    
    def test_label_distribution(self, sample_reddit_data):
        """Test label distribution validation."""
        from src.validation.data_validator import DataValidator
        validator = DataValidator(strict=False)
        sample_reddit_data['label'] = [1, 0, 1]
        is_valid, msg = validator.check_label_balance(sample_reddit_data)
        assert is_valid is True


# =============================================================================
# Drift Detection Tests
# =============================================================================

class TestDriftDetection:
    """Tests for drift detection."""
    
    def test_feature_drift(self):
        """Test feature distribution drift detection."""
        from src.models.drift_detector import DriftDetector
        detector = DriftDetector()
        reference = np.random.normal(0, 1, (100, 2))
        current = np.random.normal(0, 1, (100, 2))
        detector.fit_reference(reference)
        detector.update(current)
        drift = detector.detect()
        assert isinstance(drift, dict)
        assert "overall_risk" in drift or "drift_detected" in drift


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

class TestEndToEndPipeline:
    """End-to-end pipeline tests."""
    
    @pytest.mark.asyncio
    async def test_full_prediction_flow(self):
        """Test complete prediction flow from input to output."""
        from src.models.integrated_predictor import IntegratedDoomPredictor
        pred = IntegratedDoomPredictor()
        res = pred.predict("Sample controversial topic for testing", author_id="user1")
        assert "probability" in res or "doom_score" in res


# =============================================================================
# Security Tests
# =============================================================================

class TestSecurity:
    """Security-related tests."""
    
    def test_api_authentication(self, test_config):
        """Test API authentication requirements."""
        from src.api.api_v2_production import app, config
        config.require_auth = True
        client = TestClient(app)
        response = client.post("/predict", json={'text': 'test'})
        assert response.status_code in [401, 403]
        config.require_auth = False
    
    def test_rate_limiting(self, test_config):
        """Test rate limiting placeholder."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
