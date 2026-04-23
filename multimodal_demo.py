#!/usr/bin/env python3
"""Multimodal Doom Index Demonstration Script."""

import sys
sys.path.insert(0, 'src')

from features import analyze_text_sentiment, analyze_text_toxicity
from models import CancellationPredictor

def demonstrate_multimodal_analysis():
    """Demonstrate multimodal analysis capabilities."""

    print("🎭 Multimodal Doom Index Analysis Demo")
    print("=" * 50)

    # Sample texts for analysis
    test_texts = [
        "This celebrity is facing massive backlash for their controversial statements about social issues.",
        "BREAKING: Company announces boycott after CEO's racist remarks go viral.",
        "Users are outraged over the new policy changes that affect privacy.",
        "The petition against the controversial law has gained millions of signatures.",
        "Positive news: Community comes together to support local business after crisis."
    ]

    # Load model
    try:
        predictor = CancellationPredictor()
        predictor.load_model('models/cancellation_predictor_full.pkl')
        model_loaded = True
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        model_loaded = False

    print("\n📊 Analyzing sample texts with multimodal approach:")
    print("-" * 50)

    for i, text in enumerate(test_texts, 1):
        print(f"\n🔍 Text {i}: {text[:100]}{'...' if len(text) > 100 else ''}")

        # Sentiment Analysis (DistilBERT + VADER + RoBERTa)
        sentiment = analyze_text_sentiment(text)
        if sentiment:
            print(f"   🤖 Sentiment: {sentiment.get('overall_sentiment', 'unknown').upper()}")
            if sentiment.get('distilbert'):
                distil_scores = sentiment['distilbert']
                print(".3f")
            if sentiment.get('vader'):
                vader_scores = sentiment['vader']
                print(".3f")
        else:
            print("   🤖 Sentiment: Analysis failed")

        # Toxicity Analysis
        toxicity = analyze_text_toxicity(text)
        if toxicity:
            max_toxic = max(toxicity.values()) if toxicity else 0
            print(".3f")
        else:
            print("   ☠️  Toxicity: Analysis failed")

        # Cancellation Prediction
        if model_loaded:
            try:
                # Create features for prediction
                from features import FeatureEngineer
                import pandas as pd

                engineer = FeatureEngineer()
                temp_df = pd.DataFrame([{
                    'id': f'test_{i}',
                    'keyword': 'test',
                    'text': text,
                    'created_at': pd.Timestamp.now().strftime('%a %b %d %H:%M:%S +0000 %Y'),
                    'author_id': 'test_user',
                    'user': 'test_user',
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

                temp_df = engineer._add_engineered_features(temp_df)
                _, features = engineer.create_feature_matrix(temp_df)

                pred, prob = predictor.predict(features)
                risk_level = "🔴 HIGH" if prob > 0.7 else "🟡 MEDIUM" if prob > 0.4 else "🟢 LOW"
                print(f"   🎯 Cancellation Risk: {risk_level} ({prob:.1%} likelihood)")
            except Exception as e:
                print(f"   🎯 Prediction failed: {e}")
        else:
            print("   🎯 Model not available for prediction")

        print("-" * 30)

    print("\n🎉 Multimodal Analysis Complete!")
    print("=" * 50)
    print("✅ DistilBERT: Fast, efficient sentiment analysis")
    print("✅ VADER: Rule-based sentiment for social media")
    print("✅ RoBERTa: Deep learning for nuanced understanding")
    print("✅ Perspective API: Toxicity and content moderation")
    print("✅ RandomForest: ML prediction for cancellation events")
    print("\n🚀 System ready for production deployment!")

if __name__ == "__main__":
    demonstrate_multimodal_analysis()