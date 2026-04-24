# Doom Index Model Card

## Model Details

- **Model Name**: Doom Index v2.0 (Multimodal)
- **Version**: 2.0.0
- **Organization**: Predictive Social Risk Research Group
- **Model Date**: 2026-04-24
- **Model Type**: Multimodal Neural Network (NLP + GNN + Vision)
- **Architecture**: 
  - Text: DistilBERT-base-uncased (fine-tuned)
  - Graph: GraphSAGE + GAT ensemble on Neo4j user interaction graphs
  - Vision: CLIP ViT-B/32 for image/meme analysis
  - Fusion: Cross-modal attention with gating mechanism
  - Temporal: TGN (Temporal Graph Network) for time-evolving graphs
- **License**: MIT (Research Use)
- **Contact**: doom-index-team@university.edu

## Intended Use

### Primary Use Cases
- **Risk Assessment**: Predict the likelihood of social media backlash/cancellation events for public figures based on their posts and network context.
- **Mental Health Early Warning**: Identify users showing patterns associated with high-risk social exposure (with appropriate consent and clinical oversight).
- **Research**: Study information cascades, echo chamber formation, and outrage dynamics in online social networks.

### Out-of-Scope Uses
- **Do not use** for automated content moderation decisions without human review.
- **Do not use** to target, harass, or dox individuals.
- **Do not use** as the sole basis for employment, legal, or financial decisions.
- **Do not use** on private or encrypted communications.

## Training Data

### Data Sources
- **Reddit**: Pushshift.io archives (2008-2026), ~1.1M posts filtered to ~450K cancellation-related discussions
- **Twitter/X**: Public API samples, ~200K tweets with engagement metrics
- **Instagram**: Public profile metadata (no private data)
- **Synthetic**: LLM-generated augmentation for rare cancellation events (~50K samples)

### Data Processing
- **Anonymization**: All usernames hashed with SHA-256 + salt; PII removed via NER
- **Language Filtering**: English (en), Hindi (hi), Hinglish (hinglish) via XLM-RoBERTa language detection
- **Temporal Split**: Training (80%), Validation (10%), Test (10%) - strict time-based split to prevent leakage
- **Weak Labeling**: Multivariate heuristic combining engagement velocity, sentiment polarity, and keyword presence; LLM-validated subset (5K samples)
- **Balancing**: SMOTE + undersampling for 1:3 positive:negative ratio

### Data Statistics
| Metric | Value |
|--------|-------|
| Total Samples | 520,000 |
| Positive (Cancellation) | 130,000 |
| Negative | 390,000 |
| Avg Text Length | 145 tokens |
| Languages | en (78%), hi (12%), mixed (10%) |
| Time Span | 2018-01 to 2026-03 |

## Evaluation

### Metrics
- **AUC-ROC**: 0.918 (test set)
- **F1-Score (Macro)**: 0.874
- **F1-Score (High-Risk Class)**: 0.867
- **Precision@K**: P@10=0.91, P@100=0.83
- **Calibration ECE**: 0.032 (well-calibrated)

### Benchmarks
- **Baseline (RandomForest)**: AUC=0.84, F1=0.72
- **Text-Only (DistilBERT)**: AUC=0.89, F1=0.81
- **Graph-Only (GraphSAGE)**: AUC=0.82, F1=0.75
- **Multimodal (Full)**: AUC=0.918, F1=0.874

### Fairness Evaluation
| Protected Attribute | Disparate Impact | Equalized Odds | Status |
|---------------------|------------------|----------------|--------|
| Language (en vs hi) | 0.87 | TPR diff: 0.04 | ⚠️ Monitor |
| User Type (verified) | 0.92 | TPR diff: 0.03 | ✅ Pass |
| Region (IN vs US) | 0.85 | TPR diff: 0.06 | ⚠️ Monitor |

## Ethical Considerations

### Potential Risks
1. **Surveillance**: Could be misused for mass monitoring of dissent.
2. **Bias**: May overpredict risk for non-English speakers and marginalized communities.
3. **Chilling Effect**: Users may self-censor if aware of risk scoring.
4. **Adversarial Gaming**: Bad actors could use attack simulator to optimize harmful content.

### Mitigations
- **Differential Privacy**: Training with ε=1.0 noise to prevent memorization.
- **Federated Learning**: Simulated decentralized training to reduce data centralization.
- **Transparency**: Model cards, SHAP explanations, and open-source release.
- **Human-in-the-Loop**: All high-risk predictions flagged for human review.
- **Rate Limiting**: API throttling to prevent bulk surveillance.

## Caveats and Recommendations

### Known Limitations
- **Temporal Validity**: Model trained on 2018-2026 data; platform dynamics shift rapidly.
- **Platform Bias**: Trained primarily on Reddit/Twitter; may not generalize to TikTok, Discord, or emerging platforms.
- **Context Blindness**: Cannot detect sarcasm, cultural nuance, or rapidly evolving slang.
- **Graph Sparsity**: GNN performance degrades for users with <5 interactions.

### Recommendations
- **Retrain quarterly** with fresh data to maintain accuracy.
- **Monitor drift** weekly using KS tests and autoencoder reconstruction error.
- **Audit fairness** monthly across language and demographic groups.
- **Version control** all model artifacts with DVC.
- **A/B test** all model updates before full deployment.

## Deployment

### Infrastructure
- **Training**: H100 cluster (4x H100, CUDA 12.x) via SLURM
- **Inference**: ONNX Runtime on Kubernetes (p99 latency <50ms)
- **Streaming**: Kafka + Apache Beam for real-time processing
- **Feature Store**: Redis (online) + Parquet (offline)
- **Monitoring**: Prometheus + Grafana + custom drift detection

### API Endpoints
- `POST /analyze` - Single post analysis
- `POST /predict/batch` - Batch prediction (max 1000)
- `POST /attack/simulate` - Adversarial variant generation
- `GET /dashboard/leaderboard` - Risk leaderboard
- `GET /health` - Health check

## Citation

```bibtex
@software{doom_index_2026,
  title = {Doom Index: Predictive Social Risk Assessment},
  author = {Doom Index Team},
  year = {2026},
  url = {https://github.com/killer1panda/prsi}
}
```

## Changelog

### v2.0.0 (2026-04-24)
- Added multimodal fusion (text + graph + vision)
- Integrated Temporal Graph Networks (TGN)
- Added adversarial training for robustness
- Implemented fairness auditing pipeline
- Added real-time streaming via Kafka

### v1.0.0 (2026-03-01)
- Initial release with DistilBERT + GraphSAGE
- FastAPI serving layer
- Basic Docker deployment
