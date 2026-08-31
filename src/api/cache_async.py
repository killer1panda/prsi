"""
Asynchronous Production Redis Cache with XFetch Stampede Protection.
Implements probabilistic early expiration:
Δ = -β * δ * ln(rand())
to eliminate cache stampedes under high concurrency, with vectorized MGET/MSET.
"""

import math
import time
import json
import random
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class AsyncDoomCache:
    """
    High-throughput asynchronous Redis cache layer with XFetch algorithm
    and vectorized batch key lookup.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        default_ttl: int = 3600,
        key_prefix: str = "doom:v2:",
        beta: float = 1.0
    ):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.beta = beta
        self._in_memory_fallback: Dict[str, Tuple[Dict[str, Any], float, float]] = {}

    def make_key(self, text: str, user_id: Optional[str] = None, **kwargs) -> str:
        """Construct deterministic SHA-256 cache key."""
        payload = {
            "text": text.strip(),
            "user_id": user_id or "anon",
            **{k: v for k, v in sorted(kwargs.items()) if v is not None}
        }
        raw_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        return f"{self.key_prefix}{raw_hash}"

    async def get_with_xfetch(self, key: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Check cache with XFetch probabilistic early expiration.
        Returns: (cached_value, needs_recompute)
        """
        if self.redis is not None:
            try:
                pipe = self.redis.pipeline()
                pipe.get(key)
                pipe.ttl(key)
                val, ttl = await pipe.execute()

                if val is None:
                    return None, True

                data = json.loads(val)
                delta = data.get("_computation_cost_sec", 0.05)

                # XFetch condition: if ttl - (beta * delta * ln(random())) <= 0 -> recompute
                if ttl > 0 and (ttl - (self.beta * delta * math.log(random.random()))) <= 0:
                    return data["result"], True

                return data["result"], False
            except Exception as e:
                logger.warning(f"Redis get_with_xfetch failed: {e}. Falling back to memory.")

        # In-memory fallback
        if key in self._in_memory_fallback:
            res, exp_time, cost = self._in_memory_fallback[key]
            now = time.time()
            if now > exp_time:
                del self._in_memory_fallback[key]
                return None, True
            ttl = exp_time - now
            if (ttl - (self.beta * cost * math.log(random.random()))) <= 0:
                return res, True
            return res, False

        return None, True

    async def set(
        self,
        key: str,
        result: Dict[str, Any],
        computation_cost_sec: float = 0.05,
        ttl: Optional[int] = None
    ):
        """Set cache entry with computation cost metadata for XFetch."""
        effective_ttl = ttl or self.default_ttl
        payload = {
            "result": result,
            "_computation_cost_sec": computation_cost_sec,
            "_cached_at": time.time()
        }

        if self.redis is not None:
            try:
                await self.redis.setex(key, effective_ttl, json.dumps(payload))
                return
            except Exception as e:
                logger.warning(f"Redis set failed: {e}. Writing to memory fallback.")

        # In-memory fallback
        self._in_memory_fallback[key] = (result, time.time() + effective_ttl, computation_cost_sec)

    async def mget_batch(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Vectorized multi-key batch lookup."""
        if not keys:
            return []

        if self.redis is not None:
            try:
                raw_values = await self.redis.mget(keys)
                results = []
                for val in raw_values:
                    if val:
                        results.append(json.loads(val)["result"])
                    else:
                        results.append(None)
                return results
            except Exception as e:
                logger.warning(f"Redis mget failed: {e}. Checking memory fallback.")

        # In-memory fallback
        now = time.time()
        results = []
        for k in keys:
            if k in self._in_memory_fallback:
                res, exp_time, _ = self._in_memory_fallback[k]
                if now <= exp_time:
                    results.append(res)
                else:
                    results.append(None)
            else:
                results.append(None)
        return results

    async def mset_batch(
        self,
        items: List[Tuple[str, Dict[str, Any], float]],
        ttl: Optional[int] = None
    ):
        """Vectorized pipeline batch set."""
        if not items:
            return

        effective_ttl = ttl or self.default_ttl

        if self.redis is not None:
            try:
                pipe = self.redis.pipeline()
                for key, res, cost in items:
                    payload = json.dumps({
                        "result": res,
                        "_computation_cost_sec": cost,
                        "_cached_at": time.time()
                    })
                    pipe.setex(key, effective_ttl, payload)
                await pipe.execute()
                return
            except Exception as e:
                logger.warning(f"Redis mset failed: {e}. Writing to memory.")

        # In-memory fallback
        now = time.time()
        for key, res, cost in items:
            self._in_memory_fallback[key] = (res, now + effective_ttl, cost)
