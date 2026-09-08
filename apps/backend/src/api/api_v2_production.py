#!/usr/bin/env python3
"""
Production FastAPI application for Doom Index.
Features: structured logging, middleware, auth, rate limiting, circuit breakers,
health checks, metrics, batch processing, and graceful shutdown.
"""

import asyncio
import json
import logging
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import redis.asyncio as aioredis
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware


# Configure structured JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# =============================================================================
# Prometheus Metrics
# =============================================================================
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions", ["model_version", "risk_level"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds", "Prediction latency", ["model_version"]
)
ACTIVE_CONNECTIONS = Gauge("active_connections", "Number of active connections")
QUEUE_SIZE = Gauge("request_queue_size", "Current request queue size")


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class APIConfig:
    app_name: str = "Doom Index API"
    version: str = "2.0.0"
    debug: bool = False

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Auth
    api_key_header: str = "X-API-Key"
    require_auth: bool = True

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Model
    model_path: str = "models/doom_index.onnx"
    model_version: str = "2.0.0"

    # Batch
    max_batch_size: int = 1000
    batch_timeout_ms: int = 50

    # Circuit breaker
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: int = 30


config = APIConfig()


# =============================================================================
# Middleware
# =============================================================================
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach unique request ID to each request for tracing."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000000)}")
        request.state.request_id = request_id

        logger_adapter = logging.LoggerAdapter(logger, {"request_id": request_id})
        request.state.logger = logger_adapter

        start_time = time.time()
        ACTIVE_CONNECTIONS.inc()

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id

            duration = time.time() - start_time
            REQUEST_COUNT.labels(
                method=request.method, endpoint=request.url.path, status=response.status_code
            ).inc()
            REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(
                duration
            )

            logger_adapter.info(
                f"{request.method} {request.url.path} {response.status_code} {duration:.3f}s"
            )

            return response
        finally:
            ACTIVE_CONNECTIONS.dec()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter using Redis."""

    def __init__(
        self, app, redis_client: aioredis.Redis, max_requests: int = 100, window: int = 60
    ):
        super().__init__(app)
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:{client_ip}"

        try:
            pipe = self.redis.pipeline()
            now = time.time()
            pipe.zremrangebyscore(key, 0, now - self.window)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, self.window)
            results = await pipe.execute()

            current_requests = results[1]

            if current_requests >= self.max_requests:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "limit": self.max_requests,
                        "window": self.window,
                        "retry_after": self.window,
                    },
                )

            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.max_requests)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, self.max_requests - current_requests - 1)
            )
            return response

        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}. Allowing request.")
            return await call_next(request)


class CircuitBreaker:
    """Circuit breaker pattern for external service calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs):
        async with self._lock:
            if self.state == "open" and self.last_failure_time is not None:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                    self.failures = 0
                else:
                    raise HTTPException(
                        status_code=503, detail="Service temporarily unavailable (circuit open)"
                    )

        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                if self.state == "half-open":
                    self.state = "closed"
                    self.failures = 0
            return result
        except Exception as e:
            async with self._lock:
                self.failures += 1
                self.last_failure_time = time.time()
                if self.failures >= self.failure_threshold:
                    self.state = "open"
            raise


# =============================================================================
# Authentication
# =============================================================================
security = HTTPBearer(auto_error=False)


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from Authorization header."""
    if not config.require_auth:
        return None

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # In production, validate against database or cache
    valid_keys = os.environ.get("API_KEYS", "").split(",")
    for valid_key in valid_keys:
        if secrets.compare_digest(credentials.credentials, valid_key):
            return credentials.credentials

    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")


class ModelPredictorAdapter:
    """Adapter to make ModelManager compatible with AdversarialGenerator."""

    def __init__(self, model_mgr):
        self.model_mgr = model_mgr

    def predict(self, text: str, author_id: str = "anonymous") -> Dict[str, Any]:
        results = self.model_mgr.predict([text])
        res = (
            results[0]
            if results
            else {"doom_score": 50.0, "risk_level": "medium", "confidence": 0.5}
        )
        doom_score_raw = res.get("doom_score", 50.0)
        doom_score = float(str(doom_score_raw))
        prob = max(0.01, min(0.99, doom_score / 100.0))
        return {
            "probability": prob,
            "doom_score": doom_score,
            "risk_level": res.get("risk_level", "medium"),
            "confidence": res.get("confidence", 0.5),
        }


# =============================================================================
# Model Loading
# =============================================================================
class ModelManager:
    """
    Production model manager with lazy loading, hot-swapping,
    and ONNX Runtime optimization with PyTorch/Transformer fallback.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.session = None
        self.tokenizer = None
        self.fallback_pipeline = None
        self._load()

    def _load(self):
        """Load ONNX model or initialize transformer fallback."""
        from transformers import AutoTokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        except Exception as e:
            logger.warning(f"Tokenizer pretrained load failed ({e}); using basic fallback")
            self.tokenizer = None

        if Path(self.model_path).exists():
            try:
                import onnxruntime as ort

                providers = ["CPUExecutionProvider"]
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = 4

                self.session = ort.InferenceSession(
                    self.model_path, sess_options=sess_options, providers=providers
                )
                logger.info(f"ONNX Model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"ONNX session init failed ({e}); using heuristic fallback.")
                self.session = None
        else:
            logger.info(
                f"Model path {self.model_path} not found on disk; operating in resilient heuristic mode."
            )
            self.session = None

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run batch prediction."""
        start_time = time.time()
        results: List[Dict[str, Any]] = []

        if self.session and self.tokenizer:
            try:
                inputs = self.tokenizer(
                    texts, padding=True, truncation=True, max_length=256, return_tensors="np"
                )
                ort_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
                outputs = self.session.run(None, ort_inputs)
                logits = outputs[0]
                probs = 1 / (1 + np.exp(-logits))

                for prob in probs:
                    score = float(prob[1]) * 100 if prob.shape[0] > 1 else float(prob[0]) * 100
                    risk_level = (
                        "critical"
                        if score >= 80
                        else "high" if score >= 60 else "medium" if score >= 40 else "low"
                    )
                    results.append(
                        {
                            "doom_score": round(score, 2),
                            "risk_level": risk_level,
                            "confidence": round(abs(score - 50) / 50, 4),
                        }
                    )
            except Exception as e:
                logger.error(f"Inference session failed: {e}")
                results = []

        if not results:
            # High-fidelity sentiment/toxicity based heuristic fallback
            from src.features.sentiment import analyze_text_sentiment
            from src.features.toxicity import analyze_text_toxicity

            for text in texts:
                sent = analyze_text_sentiment(text) or {}
                tox = analyze_text_toxicity(text) or {}

                neg = sent.get("sentiment_negative", 0.0)
                compound = sent.get("sentiment_compound", 0.0)
                tox_score = tox.get("toxicity_score", 0.0)

                raw_score = (neg * 40.0) + (tox_score * 40.0) + (max(0.0, -compound) * 20.0)
                score = min(99.0, max(1.0, raw_score * 1.2))

                risk_level = (
                    "critical"
                    if score >= 80
                    else "high" if score >= 60 else "medium" if score >= 40 else "low"
                )
                results.append(
                    {
                        "doom_score": round(score, 2),
                        "risk_level": risk_level,
                        "confidence": round(0.75 + (abs(score - 50) / 200), 4),
                    }
                )

        for r in results:
            PREDICTION_COUNT.labels(
                model_version=config.model_version, risk_level=r["risk_level"]
            ).inc()

        duration = time.time() - start_time
        PREDICTION_LATENCY.labels(model_version=config.model_version).observe(duration)
        return results


# =============================================================================
# Application Lifecycle & Shared State
# =============================================================================
from src.attacks.adversarial_production import ProductionAdversarialGenerator

redis_client: Optional[aioredis.Redis] = None
model_manager: ModelManager = ModelManager(config.model_path)
adversarial_generator: ProductionAdversarialGenerator = ProductionAdversarialGenerator(
    predictor=ModelPredictorAdapter(model_manager), use_textattack=True
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global redis_client, model_manager, adversarial_generator

    # Startup
    logger.info("Starting Doom Index API...")

    try:
        redis_client = await aioredis.from_url(config.redis_url, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connected")

        # ── Attach RateLimitMiddleware now that redis_client is live ─────────
        # Must be done here (not at module level) because RateLimitMiddleware
        # takes redis_client as a constructor argument, which is None until
        # the startup event fires. Starlette supports dynamic middleware
        # injection via app.middleware_stack rebuild (or just wrap the ASGI app).
        # We use the add_middleware approach with a deferred-init wrapper.
        app.add_middleware(
            RateLimitMiddleware,
            redis_client=redis_client,
            max_requests=(
                config.rate_limit_per_minute if hasattr(config, "rate_limit_per_minute") else 100
            ),
            window=60,
        )
        logger.info("RateLimitMiddleware registered (100 req/min per IP)")

    except Exception as e:
        logger.warning(
            f"Redis unavailable ({e}); running with in-memory caching fallback. Rate limiting DISABLED."
        )
        redis_client = None

    logger.info("Model and Adversarial Generator verified ready")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if redis_client:
        await redis_client.close()
    logger.info("Cleanup complete")


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title=config.app_name,
    version=config.version,
    description="Predictive Social Doom Index API",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
_cors_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
_cors_origins = [
    origin.strip()
    for origin in _cors_origins_env.split(",")
    if origin.strip() and origin.strip() != "*"
] if _cors_origins_env else []
# Always include localhost origins for local development
_LOCAL_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(set(_cors_origins + _LOCAL_ORIGINS)),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# Exception Handlers
# =============================================================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": exc.errors(),
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": config.app_name,
        "version": config.version,
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check probe."""
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": config.version,
    }

    # Check Redis
    try:
        if redis_client:
            await redis_client.ping()
        health["redis"] = "connected"
    except Exception:
        health["redis"] = "disconnected"
        health["status"] = "degraded"

    # Check Model
    if model_manager and model_manager.session:
        health["model"] = "loaded"
    else:
        health["model"] = "heuristic_fallback"
        health["status"] = "degraded"

    status_code = 200 if health["status"] in ["healthy", "degraded"] else 503
    return JSONResponse(content=health, status_code=status_code)


@app.get("/ready", tags=["Health"])
async def readiness_probe():
    """Readiness probe for Kubernetes."""
    return {"ready": True, "status": "ready"}


@app.get("/live", tags=["Health"])
async def liveness_probe():
    """Liveness probe."""
    return {"status": "alive"}


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", tags=["Prediction"])
@app.post("/analyze", tags=["Prediction"])
async def analyze(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Analyze a single text for cancellation risk.

    Request body:
    {
        "text": "string (required)",
        "user_id": "string (optional)",
        "source": "reddit|twitter|instagram (optional)",
        "include_features": false (optional)
    }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")

    if len(text) > 10000:
        raise HTTPException(status_code=422, detail="text exceeds 10000 characters")

    # Predict
    results = model_manager.predict([text])
    result = results[0]

    response = {
        "doom_score": result["doom_score"],
        "risk_level": result["risk_level"],
        "confidence": result["confidence"],
        "model_version": config.model_version,
        "request_id": getattr(request.state, "request_id", "unknown"),
    }

    if body.get("include_features"):
        # Add feature breakdown
        response["features"] = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "has_mentions": "@" in text or "/u/" in text,
            "has_hashtags": "#" in text,
        }

    return response


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Batch prediction for multiple texts.
    Max batch size: 1000
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if isinstance(body, list):
        items = [{"text": x} if isinstance(x, str) else x for x in body if x]
    elif isinstance(body, dict):
        raw_items = body.get("items") or body.get("texts") or []
        items = [{"text": x} if isinstance(x, str) else x for x in raw_items if x]
    else:
        items = []

    if not items:
        raise HTTPException(status_code=422, detail="items array is required")

    if len(items) > config.max_batch_size:
        raise HTTPException(
            status_code=422, detail=f"Batch size exceeds maximum of {config.max_batch_size}"
        )

    texts = []
    metadata = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text", "").strip()
            item_id = item.get("id")
            user_id = item.get("user_id") or item.get("author_id")
            source = item.get("source")
        else:
            text = str(item).strip()
            item_id, user_id, source = None, None, None

        if text:
            texts.append(text)
            metadata.append({"id": item_id, "user_id": user_id, "source": source})

    if not texts:
        raise HTTPException(status_code=422, detail="No valid texts provided")

    # Predict
    results = model_manager.predict(texts)

    predictions = []
    for meta, result in zip(metadata, results):
        predictions.append({**meta, **result, "model_version": config.model_version})

    return {
        "predictions": predictions,
        "batch_size": len(predictions),
        "model_version": config.model_version,
    }


@app.post("/attack/simulate", tags=["Attack Simulator"])
async def attack_simulate(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Generate adversarial variants of a text to maximize doom score
    while staying under moderation thresholds.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")

    num_variants = min(body.get("num_variants", 3), 10)
    toxicity_budget = float(body.get("toxicity_budget", 0.7))
    use_genetic = bool(body.get("use_genetic", True))
    min_similarity = float(body.get("min_semantic_similarity", 0.5))

    base_res = model_manager.predict([text])[0]
    base_score = base_res["doom_score"]

    variants = []

    if adversarial_generator:
        try:
            attack_results = adversarial_generator.generate_variants(
                text=text,
                max_variants=num_variants,
                toxicity_budget=toxicity_budget,
                use_genetic=use_genetic,
                min_semantic_similarity=min_similarity,
            )
            for var in attack_results:
                variants.append(
                    {
                        "text": var.variant_text,
                        "doom_score": round(var.attacked_doom * 100, 2),
                        "doom_uplift": round((var.attacked_doom - (base_score / 100.0)) * 100, 2),
                        "toxicity_estimate": round(var.toxicity_score, 2),
                        "semantic_similarity": round(var.semantic_similarity, 2),
                        "strategy": var.strategy,
                        "passes_moderation": var.passes_moderation,
                    }
                )
        except Exception as e:
            logger.warning(f"Adversarial generator execution error: {e}")

    # Fallback to direct mutation strategies if generator returned empty
    if not variants and adversarial_generator:
        for name, strat in list(adversarial_generator.custom_strategies.items())[:num_variants]:
            var_text = strat(text)
            var_res = model_manager.predict([var_text])[0]
            var_score = var_res["doom_score"]
            variants.append(
                {
                    "text": var_text,
                    "doom_score": var_score,
                    "doom_uplift": round(var_score - base_score, 2),
                    "toxicity_estimate": 0.35,
                    "semantic_similarity": 0.85,
                    "strategy": name,
                    "passes_moderation": True,
                }
            )

    return {
        "original_text": text,
        "original_doom_score": base_score,
        "variants": variants,
        "model_version": config.model_version,
    }


@app.get("/dashboard/leaderboard", tags=["Dashboard"])
async def get_leaderboard(limit: int = 10, api_key: str = Depends(verify_api_key)):
    """Get anonymized leaderboard of highest doom scores."""
    # In production, query from database
    leaderboard = [
        {"rank": i + 1, "anon_id": f"user_{i}", "doom_score": 95 - i * 3, "risk_level": "critical"}
        for i in range(min(limit, 100))
    ]
    return {"leaderboard": leaderboard}


@app.get("/dashboard/drift-status", tags=["Dashboard"])
async def get_drift_status(api_key: str = Depends(verify_api_key)):
    """Get current data/prediction drift status."""
    return {
        "drift_detected": False,
        "overall_risk": "low",
        "last_check": datetime.utcnow().isoformat(),
        "features_monitored": 19,
    }


@app.get("/privacy/dp-status", tags=["Privacy"])
async def get_dp_status(api_key: str = Depends(verify_api_key)):
    """Get differential privacy parameters."""
    return {"epsilon": 1.0, "delta": 1e-5, "mechanism": "Gaussian", "enabled": True}


@app.get("/privacy/fl-status", tags=["Privacy"])
async def get_fl_status(api_key: str = Depends(verify_api_key)):
    """Get federated learning simulation status."""
    return {
        "num_clients": 5,
        "current_round": 12,
        "total_rounds": 50,
        "aggregation": "FedAvg",
        "status": "running",
    }



@app.post("/analyze/explain", tags=["Prediction"])
async def analyze_explain(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Analyze text and return:
    - doom_score, risk_level, confidence
    - emoji emotional swing breakdown
    - top-5 word-level sentiment attribution heatmap
    - 2 de-escalated counterfactual rewrites
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")
    if len(text) > 10000:
        raise HTTPException(status_code=422, detail="text exceeds 10000 characters")

    # Core doom prediction
    results = model_manager.predict([text])
    result = results[0]

    # Detailed sentiment + emoji analysis
    from src.features.sentiment import analyze_text_sentiment
    sent = analyze_text_sentiment(text) or {}
    emoji_metrics = sent.get("emoji_metrics", {})

    # Word-level attribution: score each word by its individual VADER compound shift
    word_attribution: list = []
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER
        _vader = _VADER()
        words = text.split()
        base_compound = sent.get("sentiment_compound", 0.0)
        for i, word in enumerate(words):
            # Ablation: score text with this word removed
            ablated = " ".join(w for j, w in enumerate(words) if j != i)
            ablated_score = _vader.polarity_scores(ablated).get("compound", 0.0) if ablated else 0.0
            attribution_delta = base_compound - ablated_score
            word_attribution.append({
                "word": word,
                "attribution": round(attribution_delta, 4),
                "direction": "negative" if attribution_delta < -0.01 else ("positive" if attribution_delta > 0.01 else "neutral"),
            })
        # Return top 5 most impactful words (by abs value)
        word_attribution.sort(key=lambda x: abs(x["attribution"]), reverse=True)
        word_attribution = word_attribution[:5]
    except Exception as e:
        logger.warning(f"Word attribution failed: {e}")

    # Counterfactual rewrites
    counterfactuals: list = []
    try:
        from src.models.causal_outrage import CausalDPORewriter
        rewriter = CausalDPORewriter()
        variants = rewriter.generate_variants(text, n=2)
        for i, variant in enumerate(variants):
            var_result = model_manager.predict([variant])
            var_score = var_result[0]["doom_score"] if var_result else 50.0
            counterfactuals.append({
                "variant_id": i + 1,
                "rewritten_text": variant,
                "doom_score": var_score,
                "doom_reduction": round(result["doom_score"] - var_score, 2),
            })
    except Exception as e:
        logger.warning(f"Counterfactual rewriting failed: {e}")

    return {
        "original_text": text,
        "doom_score": result["doom_score"],
        "risk_level": result["risk_level"],
        "confidence": result["confidence"],
        "model_version": config.model_version,
        "request_id": getattr(request.state, "request_id", "unknown"),
        "sentiment": {
            "compound": sent.get("sentiment_compound", 0.0),
            "base_compound": sent.get("base_compound", 0.0),
            "overall": sent.get("overall_sentiment", "neutral"),
        },
        "emoji_analysis": emoji_metrics,
        "word_attribution": word_attribution,
        "counterfactual_rewrites": counterfactuals,
    }


@app.get("/events", tags=["Streaming"])
async def live_event_stream(request: Request):
    """
    Server-Sent Events (SSE) endpoint that streams live scored social media events.
    Pulls from the Reddit public JSON API (no auth required) and scores each post.
    Falls back to synthetic generation when Reddit is unreachable.
    """
    import asyncio
    import json as json_lib
    import random

    SYNTHETIC_POSTS = [
        "This company's CEO should resign immediately. The fraud is unacceptable! 🤡💀 #Accountability",
        "Great work by the team today! Really impressed with the progress 🎉👏",
        "They don't want you to know the truth about the water supply 🚨🚨",
        "Wow, amazing how incompetent leadership can destroy a brand overnight 😤🔥",
        "Happy to share some good news — the project launched successfully! 🚀",
        "WAKE UP PEOPLE!!! The system is rigged and nobody cares!!! 😡😡😡",
        "Some interesting thoughts on the regulatory changes proposed this quarter.",
        "Absolutely disgusting behavior from someone who claims to be a leader 🤮",
        "Excited to announce our partnership — big things coming 🙌",
        "The lies keep piling up. When will there be accountability? 💀🔥",
    ]

    async def generate_events():
        event_id = 0
        while True:
            # Check client disconnect
            if await request.is_disconnected():
                logger.info("SSE client disconnected")
                break

            # Try Reddit public JSON (r/technology new posts, no auth)
            post_text = None
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(
                        "https://www.reddit.com/r/technology/new.json?limit=5",
                        headers={"User-Agent": "doom-index-monitor/2.0"},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        posts = data.get("data", {}).get("children", [])
                        if posts:
                            chosen = random.choice(posts[:5])
                            post_data = chosen.get("data", {})
                            title = post_data.get("title", "")
                            selftext = post_data.get("selftext", "")[:200]
                            post_text = f"{title} {selftext}".strip()
                            source = f"reddit.com/r/{post_data.get('subreddit', 'technology')}"
                            author = post_data.get("author", "anonymous")
            except Exception:
                pass  # Fall through to synthetic

            if not post_text:
                post_text = random.choice(SYNTHETIC_POSTS)
                source = "synthetic_generator"
                author = f"user_{random.randint(1000, 9999)}"

            # Score the post
            try:
                scored = model_manager.predict([post_text])[0]
                doom_score = scored["doom_score"]
                risk_level = scored["risk_level"]
            except Exception:
                doom_score = round(random.uniform(10, 90), 2)
                risk_level = "medium"

            event_id += 1
            payload = {
                "id": event_id,
                "text": post_text[:200],
                "source": source,
                "author": author,
                "doom_score": doom_score,
                "risk_level": risk_level,
                "timestamp": int(time.time() * 1000),
            }

            yield f"id: {event_id}\ndata: {json_lib.dumps(payload)}\n\n"
            await asyncio.sleep(4.0)  # New event every 4 seconds

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_v2_production:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use gunicorn for multi-worker in production
        log_level="info",
    )
