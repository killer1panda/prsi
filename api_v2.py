"""FastAPI v2 for Doom Index — Multimodal + Attack Simulator.

Endpoints:
  POST /analyze       → Multimodal prediction (GNN + BERT)
  POST /attack        → Shadowban attack simulation
  GET  /leaderboard   → Top doom scores
  GET  /health        → System health
  GET  /              → Streamlit dashboard redirect
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.integrated_predictor import IntegratedDoomPredictor
from src.features import analyze_text_sentiment, analyze_text_toxicity

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Doom Index API v2",
    description="Multimodal Social Doom Predictor with Adversarial Simulation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Loading ───────────────────────────────────────────────────────────

predictor = None
model_loaded = False

def load_models():
    """Lazy load models on first request."""
    global predictor, model_loaded

    if predictor is not None:
        return

    try:
        predictor = IntegratedDoomPredictor(
            model_path="models/multimodal_doom/best_model.pt",
            config_path="models/multimodal_doom/model_config.pt",
        )

        # Build graph from sample data if available
        sample_path = Path("processed_sample.csv")
        if sample_path.exists():
            import pandas as pd
            df = pd.read_csv(sample_path)
            predictor.build_graph_from_posts(df)

        model_loaded = True
        logger.info("Multimodal predictor loaded successfully")

    except Exception as e:
        logger.warning(f"Could not load multimodal model: {e}")
        model_loaded = False

# ── Request/Response Models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Post text to analyze")
    author_id: str = Field(default="anonymous", description="Author identifier")
    followers: int = Field(default=0, ge=0, description="Author follower count")
    verified: bool = Field(default=False, description="Is author verified")

class AnalyzeResponse(BaseModel):
    prediction: int
    probability: float
    doom_score: int
    risk_level: str
    sentiment: Optional[Dict[str, Any]] = None
    toxicity: Optional[Dict[str, Any]] = None
    graph_embedding_norm: Optional[float] = None
    text_embedding_norm: Optional[float] = None

class AttackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Target text to attack")
    author_id: str = Field(default="anonymous", description="Author identifier")
    max_variants: int = Field(default=5, ge=1, le=10, description="Number of adversarial variants")
    toxicity_budget: float = Field(default=0.7, ge=0.0, le=1.0, 
                                    description="Max allowed toxicity score")

class AttackVariant(BaseModel):
    variant_text: str
    original_doom: float
    attacked_doom: float
    doom_uplift: float
    toxicity_score: float
    strategy: str

class AttackResponse(BaseModel):
    original_text: str
    original_doom: float
    variants: List[AttackVariant]
    best_variant_index: int

class LeaderboardEntry(BaseModel):
    author_id: str
    doom_score: float
    risk_level: str
    followers: int

# ── Endpoints ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_models()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard or redirect."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Doom Index v2</title>
        <meta http-equiv="refresh" content="0; url=http://localhost:8501" />
        <style>
            body { font-family: monospace; background: #0a0a0a; color: #ff4444; 
                   display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
            .box { text-align: center; border: 2px solid #ff4444; padding: 40px; border-radius: 8px; }
            h1 { font-size: 2.5em; margin: 0 0 10px 0; }
            p { color: #888; }
            a { color: #ff4444; }
        </style>
    </head>
    <body>
        <div class="box">
            <h1>🔥 DOOM INDEX v2</h1>
            <p>Multimodal Social Doom Predictor</p>
            <p>Redirecting to <a href="http://localhost:8501">Dashboard</a>...</p>
            <p style="font-size: 0.8em; margin-top: 20px;">
                API: <a href="/docs">/docs</a> | 
                Health: <a href="/health">/health</a> | 
                Leaderboard: <a href="/leaderboard">/leaderboard</a>
            </p>
        </div>
    </body>
    </html>
    """

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Analyze text for cancellation risk using multimodal model."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        # Get sentiment and toxicity
        sentiment = analyze_text_sentiment(request.text)
        toxicity = analyze_text_toxicity(request.text)

        # Multimodal prediction
        result = predictor.predict(
            text=request.text,
            author_id=request.author_id,
            followers=request.followers,
            verified=request.verified,
        )

        return AnalyzeResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            doom_score=result['doom_score'],
            risk_level=result['risk_level'],
            sentiment=sentiment,
            toxicity=toxicity,
            graph_embedding_norm=result.get('graph_embedding_norm'),
            text_embedding_norm=result.get('text_embedding_norm'),
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/attack", response_model=AttackResponse)
async def attack_simulate(request: AttackRequest):
    """Generate adversarial variants that maximize doom score while evading moderation."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        # Get original doom score
        original = predictor.predict(
            text=request.text,
            author_id=request.author_id,
        )

        # Generate variants using simple strategies
        # (Full TextAttack integration can be added later)
        variants = generate_adversarial_variants(
            request.text,
            original['probability'],
            request.max_variants,
            request.toxicity_budget,
            predictor,
            request.author_id,
        )

        best_idx = max(range(len(variants)), key=lambda i: variants[i].attacked_doom)

        return AttackResponse(
            original_text=request.text,
            original_doom=original['probability'],
            variants=variants,
            best_variant_index=best_idx,
        )

    except Exception as e:
        logger.error(f"Attack simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Attack simulation failed: {str(e)}")

@app.get("/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(limit: int = 20):
    """Get top users by doom score."""
    # Placeholder — would query from database
    return [
        LeaderboardEntry(
            author_id=f"user_{i}",
            doom_score=95.0 - i * 3,
            risk_level="CRITICAL" if i < 5 else "HIGH",
            followers=10000 + i * 5000,
        )
        for i in range(min(limit, 20))
    ]

@app.get("/health")
async def health_check():
    """System health check."""
    import torch

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "version": "2.0.0",
        "features": {
            "multimodal_prediction": model_loaded,
            "sentiment": True,
            "toxicity": True,
            "attack_simulation": model_loaded,
            "gnn": model_loaded,
        }
    }

# ── Attack Simulation Logic ─────────────────────────────────────────────────

def generate_adversarial_variants(
    text: str,
    original_doom: float,
    max_variants: int,
    toxicity_budget: float,
    predictor: IntegratedDoomPredictor,
    author_id: str,
) -> List[AttackVariant]:
    """Generate adversarial text variants.

    Strategies:
    1. Emoji injection (increases engagement without toxicity)
    2. Passive→active voice shift
    3. Rhetorical question conversion
    4. Exaggeration amplifiers
    5. Controversy framing
    """
    import random

    strategies = [
        ("emoji_injection", lambda t: t + " " + random.choice(["😤", "💀", "🔥", "😡", "🤬", "👀"])),
        ("rhetorical_question", lambda t: t + " Don't you think?" if not t.endswith("?") else t),
        ("exaggeration", lambda t: t.replace("very", "extremely").replace("some", "many").replace("a few", "countless")),
        ("controversy_frame", lambda t: f"BREAKING: {t}" if not t.startswith("BREAKING") else t),
        ("outrage_punctuation", lambda t: t.replace(".", "!!!").replace("!", "!!!") if "." in t else t + "!!!"),
        ("call_to_action", lambda t: t + " Retweet if you agree."),
        ("authority_challenge", lambda t: t + " The establishment doesn't want you to know this."),
    ]

    variants = []
    used_strategies = set()

    for strategy_name, transform in strategies:
        if len(variants) >= max_variants:
            break
        if strategy_name in used_strategies:
            continue

        try:
            variant_text = transform(text)
            if variant_text == text:
                continue

            result = predictor.predict(variant_text, author_id)

            # Simple toxicity proxy (would use Perspective API in production)
            toxicity_proxy = min(1.0, original_doom + 0.1)

            if toxicity_proxy > toxicity_budget:
                continue  # Skip if over toxicity budget

            variants.append(AttackVariant(
                variant_text=variant_text,
                original_doom=original_doom,
                attacked_doom=result['probability'],
                doom_uplift=result['probability'] - original_doom,
                toxicity_score=toxicity_proxy,
                strategy=strategy_name,
            ))

            used_strategies.add(strategy_name)

        except Exception as e:
            logger.debug(f"Strategy {strategy_name} failed: {e}")
            continue

    # Sort by doom uplift descending
    variants.sort(key=lambda v: v.doom_uplift, reverse=True)

    return variants


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
