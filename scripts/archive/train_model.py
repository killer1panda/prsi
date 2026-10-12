from src.models import train_cancellation_model

# Train model on processed sample
predictor, results = train_cancellation_model(
    '/Users/ajay/Downloads/doom-index/processed_sample.csv',
    model_path='/Users/ajay/Downloads/doom-index/models/cancellation_predictor.pkl'
)

print("Training results:")
print(f"Model: {results['model_type']}")
print(f"Train size: {results['train_size']}, Test size: {results['test_size']}")
print(f"CV Scores: {results['cross_val_scores']}")
print("Classification Report:")
print(results['classification_report'])