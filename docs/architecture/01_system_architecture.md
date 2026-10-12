# System Architecture

## Overview
**PRSI (Predictive Real-time Social Intelligence)** is an enterprise-grade AI system architected for sub-second social media risk analysis, outrage cascade prediction, and de-escalation recommendation. The platform is designed around a decoupled, microservice-based architecture capable of running in lightweight local developer mode or distributed high-performance computing (HPC) clusters.

---

## 🏗️ Architectural Topology

```mermaid
graph TD
    User([Clients: Web / Desktop / Mobile]) -->|TLS / HTTPS| Kong[Kong API Gateway :8000]
    
    subgraph Gateway_Cluster [Ingress & Gateway Layer]
        Kong -->|Route /api/v2| FastAPI[FastAPI v2 Production Core :8000]
        Kong -->|Route /events| SSEStream[SSE Event Broadcaster]
    end

    subgraph Service_Mesh [Inference & Feature Services]
        FastAPI <-->|Probabilistic XFetch| Redis[(Redis 7.2 Cluster)]
        FastAPI <-->|gRPC / HTTP2| Triton[Triton Inference Server :8001]
        FastAPI <-->|Cypher Bolt| Neo4j[(Neo4j 5.x Graph DB)]
        FastAPI <-->|PyMongo| MongoDB[(MongoDB 7.0 Document Store)]
        FastAPI <-->|Cosine Search| Qdrant[(Qdrant Vector Store :6333)]
    end

    subgraph Streaming_Pipeline [High-Velocity Stream Processing]
        Kafka[Kafka Broker :9092] --> Flink[Apache Flink CEP Cluster]
        Flink -->|Alerts > threshold| FastAPI
        Kafka --> IngestionWorker[Kafka Inference Worker]
        IngestionWorker --> Redis
    end

    subgraph Defense_and_Privacy [Security & Governance]
        FastAPI --> DPAdam[Adaptive DP-Adam Engine]
        FastAPI --> CKKS[TenSEAL CKKS SMPC Aggregator]
        FastAPI --> CQR[Conformal Prediction Sentinel]
    end
```

---

## 🧩 Microservice Components

### 1. Kong API Gateway
- **Config**: Defined in [`apps/api-gateway/kong.yaml`](file:///Users/ajay/Downloads/doom-index/apps/api-gateway/kong.yaml).
- **Responsibilities**:
  - Central ingress point with TLS termination.
  - JWT / API Key verification (`Authorization: Bearer <token>`).
  - Strict rate limiting: 100 requests/minute for standard tiers, 1000 requests/minute for enterprise.
  - Path preservation (`strip_path: false`).

### 2. FastAPI Production Core
- **Source**: [`apps/backend/src/api/api_v2_production.py`](file:///Users/ajay/Downloads/doom-index/apps/backend/src/api/api_v2_production.py).
- **Key Capabilities**:
  - `/analyze`: Low-latency doom score inference.
  - `/analyze/explain`: In-depth token-level VADER attribution heatmap, emoji emotional swing extraction, and Pearlian DPO counterfactual rewrites.
  - `/attack`: Adversarial token mutation generation via GCG and AutoDAN.
  - `/events`: Real-time Server-Sent Events stream scoring social media posts dynamically.
  - Hardened CORS middleware supporting local development (`http://localhost:3000`, `http://localhost:3001`, `http://localhost:8080`).

### 3. Triton Inference Server
- **Client**: [`apps/backend/src/api/triton_client.py`](file:///Users/ajay/Downloads/doom-index/apps/backend/src/api/triton_client.py).
- **Role**: Dispatches batch model inferences for Mistral-7B and Qwen2-VL to NVIDIA TensorRT-LLM runtimes.
- **Local Fallback**: Automatically degrades to CPU-based sinusoidal projection when running locally (`HPC_MODE=0` or unreachable Triton endpoint), ensuring seamless developer workflows without CUDA hardware.

### 4. Storage & State Engines
- **Neo4j Temporal Graph**: Models user-to-user retweets, replies, mentions, and cascade topologies with dynamic edge timestamps.
- **MongoDB**: Stores raw crawled social records, metadata, author follower distributions, and cancellation event records.
- **Qdrant**: Manages 4096-dimensional Mistral-7B semantic embeddings for fast nearest-neighbor counterfactual retrieval.
- **Redis Cluster**: Caches predictions using the **XFetch probabilistic early recomputation algorithm**, preventing cache stampedes under flash traffic.

---

## 🔄 Request Lifecycle: `/analyze/explain`

```mermaid
sequenceDiagram
    autonumber
    actor Client as Web / Client App
    participant Gateway as Kong Gateway
    participant API as FastAPI Core
    participant Cache as Redis (XFetch)
    participant Model as Frontier Multimodal Core
    participant Causal as Pearlian SCM / DPO

    Client->>Gateway: POST /analyze/explain (Text, Auth Token)
    Gateway->>API: Route Request (Validated Bearer Key)
    API->>Cache: Query Key cache:predict:<hash>
    alt Cache Hit (Fresh)
        Cache-->>API: Return Cached Result
    else Cache Miss / XFetch Early Expiration
        API->>Model: Forward Text, Extract Embeddings
        Model-->>API: Logits, Norms & Uncertainty
        API->>Causal: Word Ablation & DPO Counterfactual Variants
        Causal-->>API: Top 5 Attributions + 2 Rewrites
        API->>Cache: Store Result with Probabilistic TTL
    end
    API-->>Gateway: 200 OK (JSON Payload)
    Gateway-->>Client: Render Doom Score, Heatmap & Rewrites
```
