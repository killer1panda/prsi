# NOTE: This is the v1 API using RandomForest.
# The v2 multimodal API is in api_v2.py.
# Use api_v2.py for production - it supports GraphSAGE + DistilBERT.

"""FastAPI application for doom-index predictions."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pickle
import logging

from src.features import analyze_text_sentiment, analyze_text_toxicity
from src.models import CancellationPredictor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Doom Index API",
    description="API for analyzing cancellation events and social media sentiment",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
try:
    predictor = CancellationPredictor()
    predictor.load_model('models/cancellation_predictor_full.pkl')
    model_loaded = True
except Exception as e:
    logger.warning(f"Could not load model: {e}")
    model_loaded = False

class TextAnalysisRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    sentiment: Optional[Dict[str, Any]] = None
    toxicity: Optional[Dict[str, Any]] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML."""
    try:
        with open("frontend.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Could not read frontend.html: {e}")
        raise HTTPException(status_code=500, detail="Could not read frontend.html")

@app.post("/analyze", response_model=PredictionResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text for cancellation likelihood and sentiment."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        # Get sentiment
        sentiment = analyze_text_sentiment(request.text)

        # Get toxicity
        toxicity = analyze_text_toxicity(request.text)

        # Create feature vector using proper feature engineering
        from src.features import FeatureEngineer
        engineer = FeatureEngineer()

        # Create a temporary dataframe for feature engineering
        import pandas as pd
        temp_df = pd.DataFrame([{
            'id': 'api_request',
            'keyword': 'api',
            'text': request.text,
            'created_at': pd.Timestamp.now().strftime('%a %b %d %H:%M:%S +0000 %Y'),
            'author_id': 'api_user',
            'user': 'api_user',
            'followers': 0,
            'verified': False,
            'likes': 0,
            'retweets': 0,
            'replies': 0,
            'quotes': 0,
            'hashtags': '',
            'media_count': 0,
            'media_urls': '[]',
            'is_retweet': False,
            'is_quote': False
        }])

        # Set parsed sentiment columns
        vader = sentiment.get('vader', {})
        temp_df['sentiment_compound'] = vader.get('compound', 0.0)
        temp_df['sentiment_pos'] = vader.get('pos', 0.0)
        temp_df['sentiment_neg'] = vader.get('neg', 0.0)
        temp_df['sentiment_neu'] = vader.get('neu', 0.0)
        temp_df['sentiment_text_length'] = len(request.text)

        toxicity_data = toxicity or {}
        temp_df['toxicity_toxicity'] = toxicity_data.get('toxicity', 0.0)

        # Engineer features
        temp_df = engineer._add_engineered_features(temp_df)

        # Create feature matrix
        features, _ = engineer.create_feature_matrix(temp_df)
        print(f"Features shape: {features.shape}")

        # Predict
        pred, prob = predictor.predict(features)

        return PredictionResponse(
            prediction=pred,
            probability=prob,
            sentiment=sentiment,
            toxicity=toxicity
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "features": {
            "sentiment": True,
            "toxicity": True,
            "prediction": model_loaded
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)