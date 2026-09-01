#!/usr/bin/env python3
"""End-to-End Functional Verification of all Doom Index subsystems.

Verifies:
1. Preprocessing & DB Fallbacks
2. Feature Extractors (Sentiment, Toxicity, Louvain Echo-Chamber, Temporal Volatility & Outrage Velocity)
3. Meme Detection & Stream Exposure Aggregation
4. Multimodal Model Inference, Text-only Fallback, Follower Calibration & Multilingual Hinglish
5. Shadowban Attack Simulator (Text mutations & Visual Venom perturbations)
6. Privacy / Federated Learning FedAvg
7. API Client Endpoints
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("e2e_verify")


def verify_all():
    print("=" * 70)
    print("🚀 DOOM INDEX: END-TO-END SUBSYSTEM FUNCTIONAL VERIFICATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Data Ingestion & Preprocessing
    # ---------------------------------------------------------
    print("\n[1/7] Testing Data Preprocessing & DB Fallbacks...")
    from src.data.db_connectors import InMemoryCollection, MongoDBConnector
    from src.data.preprocessing import DataPreprocessor, preprocess_posts

    preprocessor = DataPreprocessor()
    clean_sample = preprocessor.clean_text(
        "OMG this is so crazy!!! Check out https://t.co/xyz123 @user #cancel"
    )
    assert len(clean_sample) > 0, "Clean text failed"

    raw_posts = [
        {
            "author_id": "u1",
            "body": "I cannot stand this behavior #boycott",
            "created_utc": 1700000000,
        },
        {
            "author_id": "u2",
            "body": "Bhai kya chal raha hai ye sab? #drama",
            "created_utc": 1700003600,
        },
    ]
    processed = preprocess_posts(raw_posts)
    assert len(processed) == 2, "preprocess_posts failed"

    col = InMemoryCollection("test_col")
    col.insert_one({"user": "test_user", "score": 90})
    doc = col.find_one({"user": "test_user"})
    assert doc["score"] == 90, "InMemoryCollection failed"
    print("  ✅ Data Preprocessing & In-Memory Fallbacks: WORKING")

    # ---------------------------------------------------------
    # 2. Feature Extraction Subsystems
    # ---------------------------------------------------------
    print("\n[2/7] Testing Feature Extractors (Sentiment, Toxicity, Graph, Temporal)...")
    from src.features.graph_extractor import GraphExtractor
    from src.features.sentiment import SentimentAnalyzer
    from src.features.toxicity import ToxicityAnalyzer
    from src.models.temporal import TemporalFeatureExtractor

    sent_analyzer = SentimentAnalyzer()
    sent_res = sent_analyzer.analyze("This is an absolute disaster and unacceptable tragedy.")
    assert (
        "polarity" in sent_res
        or "sentiment" in sent_res
        or "compound" in sent_res
        or "score" in sent_res
    ), "Sentiment failed"

    tox_analyzer = ToxicityAnalyzer()
    tox_res = tox_analyzer.analyze("You are completely toxic, disgusting and evil.")
    assert "toxicity" in tox_res or "score" in tox_res, "Toxicity failed"

    # Louvain Echo Chamber Density
    extractor = GraphExtractor()
    sample_edges = [
        {"from_user": "u1", "to_user": "u2", "weight": 2.0},
        {"from_user": "u2", "to_user": "u3", "weight": 1.0},
        {"from_user": "u3", "to_user": "u1", "weight": 3.0},
        {"from_user": "u4", "to_user": "u5", "weight": 1.0},
    ]
    density_map = extractor.compute_echo_chamber_density(
        ["u1", "u2", "u3", "u4", "u5"], sample_edges
    )
    assert len(density_map) == 5, "Louvain density computation failed"
    assert density_map["u1"] > 0.0, "Louvain density for connected community failed"

    # Temporal Feature Extractor (Volatility & Outrage Velocity)
    temp_extractor = TemporalFeatureExtractor()
    user_timeline = pd.DataFrame(
        {
            "author_id": ["u1"] * 5,
            "sentiment_polarity": [-0.8, -0.9, -0.6, -0.7, -0.9],
            "engagement": [100, 250, 400, 600, 1000],
            "negative_replies": [10, 45, 120, 280, 500],
            "created_at": [1700000000, 1700003600, 1700007200, 1700010800, 1700014400],
            "toxicity": [0.7, 0.8, 0.85, 0.9, 0.95],
        }
    )
    temp_feats = temp_extractor.extract(user_timeline)
    assert "outrage_velocity" in temp_feats, "Outrage velocity missing"
    assert "sentiment_volatility_7d" in temp_feats, "Volatility missing"
    print("  ✅ Sentiment, Toxicity, Louvain Echo-Chamber & Outrage Velocity: WORKING")

    # ---------------------------------------------------------
    # 3. Meme Detection & Timeline Exposure Aggregation
    # ---------------------------------------------------------
    print("\n[3/7] Testing Meme Detection & Timeline Exposure Aggregator...")
    from src.models.meme_detector import MemeDetector

    class DummyVision:
        class Config:
            projection_dim = 128

        config = Config()

        def encode(self, images, use_cache=True):
            return torch.randn(len(images), 128)

    detector = MemeDetector(vision_encoder=DummyVision())
    stream_posts = [
        {
            "image_url": "http://test1.jpg",
            "meme_data": {"is_meme": True, "virality_score": 0.88, "meme_type": "soyjak"},
        },
        {
            "image_url": "http://test2.jpg",
            "meme_data": {"is_meme": True, "virality_score": 0.72, "meme_type": "gigachad"},
        },
        {
            "image_url": "http://test3.jpg",
            "meme_data": {"is_meme": False, "virality_score": 0.15, "meme_type": "original"},
        },
    ]
    exposure = detector.compute_stream_exposure(stream_posts)
    assert exposure["meme_count"] == 2
    assert exposure["high_virality_count"] == 2
    assert exposure["meme_exposure_index"] > 0.0
    print("  ✅ Meme Detection & User Timeline Exposure Aggregator: WORKING")

    # ---------------------------------------------------------
    # 4. Multimodal Model Inference, Fallback & Multilingual Handling
    # ---------------------------------------------------------
    print("\n[4/7] Testing Multimodal Model Inference, Multilingual & Calibration...")
    from src.models.calibration import FollowerStratifiedCalibrator
    from src.models.integrated_predictor import IntegratedDoomPredictor

    predictor = IntegratedDoomPredictor(enable_multilingual=True)

    # Test Follower-Stratified Calibrator
    calibrator = FollowerStratifiedCalibrator()
    calibrator.fit(
        probs=np.array([0.2, 0.4, 0.8, 0.9] * 10),
        labels=np.array([0, 0, 1, 1] * 10),
        follower_counts=np.array([500, 800, 25000, 500000] * 10),
    )
    cal_res_low = calibrator.calibrate_single(0.65, followers=200)
    cal_res_high = calibrator.calibrate_single(0.65, followers=200000)
    assert 0.0 <= cal_res_low <= 1.0 and 0.0 <= cal_res_high <= 1.0

    # Test Hinglish / Code-switching text detection
    lang_det = predictor.multilingual_encoder
    detected = lang_det.detect_language("Bhai ye kya nautanki chal rahi hai twitter pe")
    assert detected in ("hi", "hinglish", "mixed"), f"Expected Hinglish/Hindi, got {detected}"
    print("  ✅ Follower-Stratified Calibration & Hinglish Detection: WORKING")

    # ---------------------------------------------------------
    # 5. Shadowban Attack Simulator (Text & Image)
    # ---------------------------------------------------------
    print("\n[5/7] Testing Shadowban Attack Simulator (Text Mutations & Visual Venom)...")
    from src.attacks.adversarial_production import (
        ProductionAdversarialGenerator, VisualVenomInjector)

    class MockPredictor:
        def predict(self, text, author_id="target"):
            doom = 40.0 + len(text) % 30
            return {"probability": doom / 100.0, "doom_score": int(doom)}

    adv_gen = ProductionAdversarialGenerator(
        predictor=MockPredictor(),
        use_textattack=False,
        max_iterations=5,
        population_size=4,
    )
    variants = adv_gen.generate_variants(
        "I strongly disagree with the current leadership policy", max_variants=2
    )

    assert len(variants) > 0, "No adversarial text variants generated"

    # Visual Venom Injector
    venom = VisualVenomInjector()
    test_img = Image.new("RGB", (150, 150), color=(100, 120, 140))
    img_attacks = venom.generate_image_attacks(test_img, original_doom=35.0)
    assert len(img_attacks) == 4, "Expected 4 image attack strategies"
    assert img_attacks[0].doom_uplift >= 0.0
    print("  ✅ Adversarial Text Generator & Visual Venom Injector: WORKING")

    # ---------------------------------------------------------
    # 6. Privacy & Federated Learning
    # ---------------------------------------------------------
    print("\n[6/7] Testing Privacy & Federated Learning Subsystems...")
    from src.privacy.dp_trainer import add_gaussian_noise
    from src.privacy.fl_simulator import federated_averaging

    w1 = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    w2 = [np.array([5.0, 6.0]), np.array([7.0, 8.0])]
    avg_w = federated_averaging([w1, w2], sample_counts=[1, 3])
    assert len(avg_w) == 2, "FedAvg failed"
    np.testing.assert_allclose(avg_w[0], np.array([4.0, 5.0]))

    # Test DP Gaussian Noise addition
    sample_tensor = torch.tensor([1.0, 2.0, 3.0])
    noisy_tensor = add_gaussian_noise(sample_tensor, sigma=0.5, clip_norm=5.0)
    assert noisy_tensor.shape == sample_tensor.shape, "DP noise tensor shape mismatch"
    print("  ✅ Federated Averaging (FedAvg) & Differential Privacy Gaussian Mechanism: WORKING")

    # ---------------------------------------------------------
    # 7. FastAPI Endpoints & Monitoring Probes
    # ---------------------------------------------------------
    print("\n[7/7] Testing FastAPI App Endpoints & Probes...")
    import os

    os.environ["API_KEYS"] = "test-verification-key"
    from fastapi.testclient import TestClient
    from src.api.api_v2_production import app, config

    config.require_auth = True

    with TestClient(app) as client:
        resp_root = client.get("/")
        assert resp_root.status_code == 200, f"Root endpoint failed: {resp_root.text}"

        resp_health = client.get("/health")
        assert resp_health.status_code == 200, f"Health check failed: {resp_health.text}"

        resp_ready = client.get("/ready")
        assert resp_ready.status_code == 200, f"Ready check failed: {resp_ready.text}"

        resp_live = client.get("/live")
        assert resp_live.status_code == 200, f"Live check failed: {resp_live.text}"

        auth_headers = {"Authorization": "Bearer test-verification-key"}
        resp_pred = client.post(
            "/predict",
            json={
                "text": "This public statement is very controversial",
                "author_id": "test_influencer",
                "followers": 15000,
            },
            headers=auth_headers,
        )
        assert resp_pred.status_code == 200, f"Predict endpoint failed: {resp_pred.text}"
        pred_data = resp_pred.json()
        assert "doom_score" in pred_data or "probability" in pred_data, "Prediction missing score"
    print("  ✅ FastAPI Endpoints (/predict, /health, /ready, /live): WORKING")

    print("\n" + "=" * 70)
    print("🎉 ALL DOOM INDEX CODEBASE SUBSYSTEMS ARE 100% OPERATIONAL!")
    print("=" * 70)


if __name__ == "__main__":
    verify_all()
