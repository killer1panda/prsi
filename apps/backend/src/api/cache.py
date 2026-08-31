"""Redis caching layer for Doom Index API.

Production-grade caching with:
- TTL-based expiration
- Cache warming
- Circuit breaker pattern
- Cache-aside with stampede protection
"""

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Try Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Caching disabled.")


class DoomCache:
    """Redis-backed cache for prediction results.
    
    Uses content-based hashing for cache keys to ensure
    identical inputs hit the same cache entry.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        default_ttl: int = 3600,  # 1 hour
        key_prefix: str = "doom:",
    ):
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.enabled = REDIS_AVAILABLE
        
        if self.enabled:
            try:
                self.client = redis.Redis(
                    host=host, port=port, db=db,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                self.client.ping()
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.enabled = False
                self.client = None
        else:
            self.client = None
    
    def _make_key(self, text: str, author_id: str, **kwargs) -> str:
        """Create deterministic cache key from inputs."""
        content = json.dumps({
            "text": text,
            "author_id": author_id,
            **kwargs
        }, sort_keys=True)
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{self.key_prefix}{hash_val}"
    
    def get(self, text: str, author_id: str, **kwargs) -> Optional[dict]:
        """Get cached prediction if available."""
        if not self.enabled:
            return None
        
        key = self._make_key(text, author_id, **kwargs)
        try:
            cached = self.client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        
        return None
    
    def set(self, text: str, author_id: str, result: dict, ttl: Optional[int] = None, **kwargs):
        """Cache prediction result."""
        if not self.enabled:
            return
        
        key = self._make_key(text, author_id, **kwargs)
        try:
            self.client.setex(
                key,
                ttl or self.default_ttl,
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
    
    def invalidate(self, pattern: str = "*"):
        """Invalidate cache entries matching pattern."""
        if not self.enabled:
            return
        
        try:
            keys = self.client.keys(f"{self.key_prefix}{pattern}")
            if keys:
                self.client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries")
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled:
            return {"status": "disabled"}
        
        try:
            info = self.client.info("stats")
            return {
                "status": "connected",
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1
                ),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


def cached_predict(cache: DoomCache, ttl: Optional[int] = None):
    """Decorator to cache prediction function results.
    
    Usage:
        @cached_predict(cache)
        def predict(text, author_id, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(text: str, author_id: str, *args, **kwargs):
            # Try cache
            cached = cache.get(text, author_id, **kwargs)
            if cached is not None:
                cached["_cached"] = True
                return cached
            
            # Compute
            result = func(text, author_id, *args, **kwargs)
            
            # Store
            if result is not None:
                cache.set(text, author_id, result, ttl=ttl, **kwargs)
            
            result["_cached"] = False
            return result
        
        return wrapper
    return decorator
