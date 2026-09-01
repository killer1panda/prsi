"""Unit tests for Phase 5 blueprint extras."""

import numpy as np
import pytest
import torch
from PIL import Image
from src.attacks.adversarial_production import (ImageAttackResult,
                                                VisualVenomInjector)
from src.models.calibration import (CalibrationAnalyzer,
                                    FollowerStratifiedCalibrator)
from src.models.meme_detector import MemeDetector
from src.privacy.fl_simulator import federated_averaging


def test_federated_averaging_optional_typing():
    """Test federated averaging runs cleanly without typing errors."""
    weights1 = [np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 1.0]])]
    weights2 = [np.array([3.0, 4.0]), np.array([[3.0, 0.0], [0.0, 3.0]])]

    # Test with default sample_counts (None)
    avg = federated_averaging([weights1, weights2])
    assert len(avg) == 2
    np.testing.assert_allclose(avg[0], np.array([2.0, 3.0]))

    # Test with custom sample counts
    avg_weighted = federated_averaging([weights1, weights2], sample_counts=[3, 1])
    np.testing.assert_allclose(avg_weighted[0], np.array([1.5, 2.5]))


def test_follower_stratified_calibrator():
    """Test follower-stratified calibration head across low/mid/high reach."""
    calibrator = FollowerStratifiedCalibrator(low_threshold=1000, high_threshold=50000)

    # Generate synthetic probs, labels, follower counts
    np.random.seed(42)
    probs = np.random.uniform(0.1, 0.9, 100)
    labels = (probs > 0.5).astype(int)
    followers = np.array([100] * 35 + [10000] * 35 + [500000] * 30)

    calibrator.fit(probs, labels, followers)
    assert calibrator.is_fitted

    # Test single calibration for each tier
    low_cal = calibrator.calibrate_single(0.65, followers=500)
    mid_cal = calibrator.calibrate_single(0.65, followers=5000)
    high_cal = calibrator.calibrate_single(0.65, followers=1000000)

    assert 0.0 <= low_cal <= 1.0
    assert 0.0 <= mid_cal <= 1.0
    assert 0.0 <= high_cal <= 1.0


def test_meme_stream_exposure_aggregation():
    """Test timeline meme exposure aggregation for user streams."""

    class DummyVisionEncoder:
        class Config:
            projection_dim = 128

        config = Config()

        def encode(self, images, use_cache=True):
            return torch.randn(len(images), 128)

    detector = MemeDetector(vision_encoder=DummyVisionEncoder())

    # Mock stream with pre-computed meme detections
    posts = [
        {
            "id": "p1",
            "image_url": "http://img1.jpg",
            "meme_data": {"is_meme": True, "virality_score": 0.85, "meme_type": "drake"},
        },
        {
            "id": "p2",
            "image_url": "http://img2.jpg",
            "meme_data": {"is_meme": True, "virality_score": 0.70, "meme_type": "distracted_bf"},
        },
        {
            "id": "p3",
            "image_url": "http://img3.jpg",
            "meme_data": {"is_meme": False, "virality_score": 0.10, "meme_type": "original"},
        },
        {"id": "p4", "text": "Text only post without images"},
    ]

    exposure = detector.compute_stream_exposure(posts)
    assert exposure["total_posts"] == 4
    assert exposure["image_posts_count"] == 3
    assert exposure["meme_count"] == 2
    assert exposure["high_virality_count"] == 2
    assert exposure["meme_frequency"] == pytest.approx(2 / 3, 0.01)
    assert exposure["meme_exposure_index"] > 0.0
    assert "drake" in exposure["top_templates"]


def test_visual_venom_injector():
    """Test visual venom injector generates adversarial image variants."""
    injector = VisualVenomInjector()

    # Create synthetic test image
    img = Image.new("RGB", (200, 200), color=(120, 100, 80))

    # Test perturbation
    img_pert, l2_diff = injector.perturb_image(img, strategy="subtle_watermark", intensity=0.2)
    assert img_pert.size == img.size
    assert l2_diff >= 0.0

    # Test variant generation
    results = injector.generate_image_attacks(img, original_doom=35.0)
    assert len(results) == 4
    assert all(isinstance(r, ImageAttackResult) for r in results)
    assert all(r.attacked_doom >= r.original_doom for r in results)
    assert results[0].doom_uplift >= results[-1].doom_uplift  # Sorted descending
