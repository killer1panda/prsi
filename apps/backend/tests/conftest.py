"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_post():
    """Sample post data for testing."""
    return {
        "post_id": "test123",
        "text": "This is a test post about #drama and #cancellation",
        "author": "test_user_hash",
        "source": "twitter",
        "created_at": "2024-01-15T10:30:00",
        "metrics": {
            "likes": 100,
            "retweets": 50,
            "replies": 25,
        },
        "hashtags": ["drama", "cancellation"],
        "mentions": ["user1", "user2"],
    }


@pytest.fixture
def sample_posts():
    """Multiple sample posts for testing."""
    return [
        {
            "post_id": "test1",
            "text": "First test post",
            "author": "user1",
            "source": "twitter",
            "created_at": "2024-01-15T10:00:00",
        },
        {
            "post_id": "test2",
            "text": "Second test post",
            "author": "user2",
            "source": "reddit",
            "created_at": "2024-01-15T11:00:00",
        },
        {
            "post_id": "test3",
            "text": "Third test post",
            "author": "user1",
            "source": "instagram",
            "created_at": "2024-01-15T12:00:00",
        },
    ]
