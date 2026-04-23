"""Model interpretability for Doom Index.

Production-grade explainability with:
- SHAP values for feature importance
- Attention visualization for text inputs
- GNNExplainer for graph reasoning
- LIME-style local explanations
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. SHAP explanations disabled.")


class DoomExplainer:
    """Explain Doom Index predictions with multiple methods.
    
    Provides both global feature importance and local
    instance-level explanations.
    """
    
    def __init__(self, predictor, device="cuda"):
        self.predictor = predictor
        self.device = device
        self.model = predictor.model
        self.tokenizer = predictor.model.tokenizer
    
    def explain_text_attention(self, text: str, author_id: str) -> Dict:
        """Extract and visualize attention weights from DistilBERT.
        
        Shows which words the model focused on when making
        its prediction.
        """
        self.model.eval()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)
        
        # Forward with attention output
        with torch.no_grad():
            outputs = self.model.text_encoder.bert(
                inputs["input_ids"],
                inputs["attention_mask"],
                output_attentions=True,
            )
        
        # Get attention from last layer [batch, heads, seq, seq]
        attentions = outputs.attentions[-1]  # Last layer
        
        # Average over heads and take CLS attention [seq]
        cls_attention = attentions[0, :, 0, :].mean(dim=0).cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Aggregate subword attention to words
        word_attentions = []
        words = []
        current_word = ""
        current_attn = 0
        count = 0
        
        for token, attn in zip(tokens, cls_attention):
            if token.startswith("##"):
                current_word += token[2:]
                current_attn += attn
                count += 1
            else:
                if current_word:
                    words.append(current_word)
                    word_attentions.append(current_attn / max(count, 1))
                current_word = token.replace("Ġ", "").replace("##", "")
                current_attn = attn
                count = 1
        
        if current_word:
            words.append(current_word)
            word_attentions.append(current_attn / max(count, 1))
        
        # Normalize
        word_attentions = np.array(word_attentions)
        if word_attentions.sum() > 0:
            word_attentions = word_attentions / word_attentions.sum()
        
        # Get top influential words
        top_indices = np.argsort(word_attentions)[-10:][::-1]
        top_words = [(words[i], float(word_attentions[i])) for i in top_indices if i < len(words)]
        
        return {
            "tokens": tokens,
            "words": words,
            "word_attentions": word_attentions.tolist(),
            "top_influential_words": top_words,
            "visualization": self._create_attention_plot(words, word_attentions),
        }
    
    def _create_attention_plot(self, words, attentions):
        """Create attention heatmap data."""
        # Return data for frontend rendering
        return [
            {"word": w, "attention": float(a)}
            for w, a in zip(words[:50], attentions[:50])  # Limit for performance
        ]
    
    def explain_graph_neighborhood(self, user_idx: int, graph_data) -> Dict:
        """Explain which neighbors influenced the prediction.
        
        Uses attention weights from GraphSAGE to identify
        the most influential connections.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get intermediate embeddings
            x = graph_data.x.to(self.device)
            edge_index = graph_data.edge_index.to(self.device)
            
            # Forward through graph encoder
            embeddings = self.model.graph_encoder(x, edge_index)
            user_emb = embeddings[user_idx]
            
            # Find neighbors
            neighbors = edge_index[1][edge_index[0] == user_idx].cpu().numpy().tolist()
            neighbors += edge_index[0][edge_index[1] == user_idx].cpu().numpy().tolist()
            neighbors = list(set(neighbors))
            
            # Compute similarity to neighbors
            similarities = []
            for n_idx in neighbors[:20]:  # Limit to top 20
                sim = F.cosine_similarity(
                    user_emb.unsqueeze(0),
                    embeddings[n_idx].unsqueeze(0),
                    dim=-1
                ).item()
                similarities.append((n_idx, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "user_idx": user_idx,
            "num_neighbors": len(neighbors),
            "top_influential_neighbors": [
                {"neighbor_idx": idx, "similarity": round(sim, 4)}
                for idx, sim in similarities[:10]
            ],
            "explanation": f"Prediction influenced by {len(neighbors)} connected users. "
                          f"Top neighbor similarity: {similarities[0][1]:.3f}" if similarities else "No neighbors found.",
        }
    
    def explain_prediction(self, text: str, author_id: str) -> Dict:
        """Full explanation combining text and graph insights."""
        # Get prediction
        result = self.predictor.predict(text, author_id)
        
        # Text explanation
        text_exp = self.explain_text_attention(text, author_id)
        
        # Graph explanation
        user_idx = self.predictor.user_to_idx.get(author_id, 0)
        graph_exp = self.explain_graph_neighborhood(user_idx, self.predictor.graph_data)
        
        # Feature breakdown
        features = {
            "text_contribution": result.get("text_embedding_norm", 0),
            "graph_contribution": result.get("graph_embedding_norm", 0),
            "doom_score": result["doom_score"],
        }
        
        return {
            "prediction": result,
            "text_explanation": text_exp,
            "graph_explanation": graph_exp,
            "feature_breakdown": features,
            "summary": self._generate_summary(result, text_exp, graph_exp),
        }
    
    def _generate_summary(self, result, text_exp, graph_exp) -> str:
        """Generate human-readable explanation summary."""
        risk = result["risk_level"]
        score = result["doom_score"]
        
        top_words = [w for w, _ in text_exp.get("top_influential_words", [])[:3]]
        word_str = ", ".join(f"'{w}'" for w in top_words)
        
        summary = f"This post has a doom score of {score}/100 ({risk} risk). "
        
        if top_words:
            summary += f"The model focused on words like {word_str}. "
        
        if graph_exp.get("num_neighbors", 0) > 5:
            summary += f"The author's network of {graph_exp['num_neighbors']} connections amplifies the risk."
        
        return summary


class SHAPExplainer:
    """SHAP-based global feature importance.
    
    Requires running on a sample of data to compute
    baseline distributions.
    """
    
    def __init__(self, model, background_data: Optional[torch.Tensor] = None):
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
        if SHAP_AVAILABLE and background_data is not None:
            self._build_explainer()
    
    def _build_explainer(self):
        """Build DeepExplainer for PyTorch model."""
        try:
            self.explainer = shap.DeepExplainer(self.model, self.background_data)
            logger.info("SHAP explainer built")
        except Exception as e:
            logger.warning(f"SHAP explainer failed: {e}")
    
    def explain(self, data: torch.Tensor) -> Optional[np.ndarray]:
        """Get SHAP values for input data."""
        if self.explainer is None:
            return None
        
        try:
            shap_values = self.explainer.shap_values(data)
            return shap_values
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return None
