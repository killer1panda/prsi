# FastAPI v2 Production Reference

## Overview
The PRSI backend exposes a high-performance REST and Streaming API powered by **FastAPI**, with full OpenAPI v3 documentation available at `http://localhost:8000/docs`.

**Base URL**: `http://localhost:8000`  
**Authentication**: `Authorization: Bearer <API_KEY>` (Default local dev key: `doom_dev_key`)

---

## 📌 Endpoints

### 1. `POST /analyze`
Performs fast multimodal inference returning risk levels, continuous doom scores, and confidence.

#### Request
```json
{
  "text": "This company's management is engaging in clear corruption. Everyone needs to boycott! 😡",
  "author_id": "user_9401",
  "followers": 25000,
  "verified": true
}
```

#### Response (`200 OK`)
```json
{
  "prediction": 1,
  "probability": 0.884,
  "doom_score": 88,
  "risk_level": "RISK_LEVEL_EXTREME",
  "confidence": 0.94,
  "sentiment": {
    "compound": -0.84,
    "overall": "negative"
  },
  "toxicity": {
    "toxicity_score": 0.78
  }
}
```

---

### 2. `POST /analyze/explain`
Deep explainability endpoint providing word-level ablation attributions, emoji emotional swing breakdowns, and counterfactual rewrites.

#### Request
```json
{
  "text": "This CEO is a total FRAUD and must resign immediately! 🤡💀"
}
```

#### Response (`200 OK`)
```json
{
  "original_text": "This CEO is a total FRAUD and must resign immediately! 🤡💀",
  "doom_score": 92.4,
  "risk_level": "RISK_LEVEL_EXTREME",
  "confidence": 0.95,
  "sentiment": {
    "compound": -0.91,
    "base_compound": -0.72,
    "overall": "negative"
  },
  "emoji_analysis": {
    "emoji_count": 2,
    "outrage_score": 0.90,
    "panic_score": 0.20,
    "irony_flag": true
  },
  "word_attribution": [
    { "word": "FRAUD", "attribution": -0.42, "direction": "negative" },
    { "word": "resign", "attribution": -0.25, "direction": "negative" },
    { "word": "immediately!", "attribution": -0.15, "direction": "negative" }
  ],
  "counterfactual_rewrites": [
    {
      "variant_id": 1,
      "rewritten_text": "This CEO is facing alleged misconduct and calls for leadership change 🤔⚠️",
      "doom_score": 41.2,
      "doom_reduction": 51.2
    },
    {
      "variant_id": 2,
      "rewritten_text": "Sharing for awareness: This CEO is facing alleged misconduct and calls for leadership change 🤔⚠️",
      "doom_score": 34.5,
      "doom_reduction": 57.9
    }
  ]
}
```

---

### 3. `POST /attack`
Generates adversarial mutations to evaluate system robustness.

#### Request
```json
{
  "text": "The establishment is deceiving everyone.",
  "max_variants": 3,
  "toxicity_budget": 0.7
}
```

#### Response (`200 OK`)
```json
{
  "original_text": "The establishment is deceiving everyone.",
  "original_doom": 0.42,
  "variants": [
    {
      "variant_text": "The establishment is deceiving everyone. 😡🔥 Retweet if you agree.",
      "original_doom": 0.42,
      "attacked_doom": 0.78,
      "doom_uplift": 0.36,
      "toxicity_score": 0.52,
      "strategy": "call_to_action"
    }
  ],
  "best_variant_index": 0
}
```

---

### 4. `GET /events`
Server-Sent Events (SSE) stream pushing scored social events to frontend clients.

```bash
curl -N http://localhost:8000/events
```

```text
id: 1
data: {"id": 1, "text": "Breaking news on tech regulations", "doom_score": 24.5, "risk_level": "RISK_LEVEL_LOW", "timestamp": 1728771200000}

id: 2
data: {"id": 2, "text": "CEO under investigation for embezzlement 🚨", "doom_score": 89.1, "risk_level": "RISK_LEVEL_EXTREME", "timestamp": 1728771204000}
```

---

### 5. `GET /health`
System diagnostics and hardware accelerator check.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": false,
  "cuda_devices": 0,
  "version": "2.0.0",
  "features": {
    "multimodal_prediction": true,
    "sentiment": true,
    "toxicity": true,
    "attack_simulation": true,
    "gnn": true
  }
}
```
