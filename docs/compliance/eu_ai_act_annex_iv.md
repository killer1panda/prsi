# EU AI Act Annex IV Compliance Dossier

## Overview
Under the European Union Artificial Intelligence Act (Regulation EU 2024/1689), AI systems deployed to analyze public sentiment, predict collective social behavior, or recommend intervention measures may fall under **High-Risk AI Systems** (Annex III). PRSI incorporates compliance-as-code, generating and maintaining technical documentation conforming to **Annex IV**.

---

## 📋 Annex IV Technical Documentation Mapping

| Annex IV Requirement | PRSI Implementation | Evidence & Code Location |
| :--- | :--- | :--- |
| **Section 1: General Description** | System architecture, trilateral multimodal fusion, foundation LLM & GNN design. | [`docs/architecture/01_system_architecture.md`](file:///Users/ajay/Downloads/doom-index/docs/architecture/01_system_architecture.md) |
| **Section 2: Methods & Data Provenance** | Preprocessing, deduplication, synthetic augmentation, class balancing without target leakage. | [`apps/backend/src/data/weak_labeling.py`](file:///Users/ajay/Downloads/doom-index/apps/backend/src/data/weak_labeling.py) |
| **Section 3: Model Architecture & Training** | 4-bit QLoRA, Mistral-7B, Qwen2-VL, Hawkes GATv2, Focal Continuous Loss, DeepSpeed ZeRO-3. | [`apps/backend/src/models/multimodal_trainer.py`](file:///Users/ajay/Downloads/doom-index/apps/backend/src/models/multimodal_trainer.py) |
| **Section 4: Monitoring, Functioning & Control** | Stateful CEP cascade detection, KS/PSI drift detection, XFetch stampede protection. | [`apps/backend/src/models/drift_detector.py`](file:///Users/ajay/Downloads/doom-index/apps/backend/src/models/drift_detector.py) |
| **Section 5: Risk Management & Mitigation** | Causal DPO de-escalation rewriter, red-teaming with GCG/AutoDAN, adversarial training. | [`apps/backend/src/attacks/gcg_autodan.py`](file:///Users/ajay/Downloads/doom-index/apps/backend/src/attacks/gcg_autodan.py) |
| **Section 6: Cyber-Security & Privacy** | Layer-wise DP-Adam ($(\varepsilon, \delta)$-DP), TenSEAL CKKS Homomorphic FL, Kong API Gateway. | [`apps/backend/src/privacy/encrypted_fl.py`](file:///Users/ajay/Downloads/doom-index/apps/backend/src/privacy/encrypted_fl.py) |

---

## 📑 Automated Dossier Generation
The compliance dossier is programmatically assembled and validated in:
- **JSON Format:** [`reports/compliance/EU_AI_ACT_ANNEX_IV_DOSSIER.json`](file:///Users/ajay/Downloads/doom-index/reports/compliance/EU_AI_ACT_ANNEX_IV_DOSSIER.json)
- **PDF Report:** [`reports/RED_TEAM_SECURITY_AUDIT_REPORT.pdf`](file:///Users/ajay/Downloads/doom-index/reports/RED_TEAM_SECURITY_AUDIT_REPORT.pdf)
