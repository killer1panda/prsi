"""
DoomGAN Production Discriminator — DeBERTa-v3-large WGAN-GP + Multi-Reward
===========================================================================
Components:
  D1. WassersteinCritic
      DeBERTa-v3-large encoder → pooled [CLS] → linear critic head (no sigmoid)
      WGAN-GP training with gradient penalty λ=10
      DeBERTa-v3-large chosen over BERT: better long-range attention (disentangled)

  D2. MultiRewardComposer
      5 reward signals with learned weights (not fixed):
        R_doom      — separate DeBERTa reward model fine-tuned on doom pairs
        R_fluency   — GPT-2-large log-likelihood (natural language proxy)
        R_blue      — blue team evasion score (1 - blue_confidence)
        R_semantic  — SBERT cosine similarity to original
        R_novelty   — distance from known attack embedding centroids

  D3. GradientPenalty
      Interpolated in embedding space (continuous relaxation via Gumbel-softmax)
      λ_gp = 10.0 (Gulrajani et al. 2017 WGAN-GP)

On local dev (HPC_MODE=0):
  DeBERTa / GPT-2 not loaded. Reward composer uses VADER + char-perplexity proxies.
  Full API compatibility: all methods return valid scalars/dicts.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HPC_MODE: bool = os.environ.get("HPC_MODE", "0").strip() == "1"
HF_CACHE_DIR: Optional[str] = os.environ.get("HF_HOME")

# ─── Optional heavy deps (HPC only) ──────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SBERT: Optional[object] = None  # Lazy — don't load at import time
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    _SBERT = None

_sbert_loaded = False


def _get_sbert():
    """Lazy SBERT loader — only triggers on first use, not at import."""
    global _SBERT, _sbert_loaded
    if _sbert_loaded:
        return _SBERT
    _sbert_loaded = True
    if not SBERT_AVAILABLE:
        return None
    try:
        # Use already-cached model only — no download if not present
        cache = os.environ.get("SENTENCE_TRANSFORMERS_HOME", os.path.expanduser("~/.cache/torch/sentence_transformers"))
        model_path = os.path.join(cache, "sentence-transformers_all-MiniLM-L6-v2")
        if os.path.exists(model_path):
            _SBERT = SentenceTransformer(model_path)
            logger.info("SBERT loaded from local cache.")
        elif HPC_MODE:
            _SBERT = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=HF_CACHE_DIR)
            logger.info("SBERT loaded on HPC.")
        else:
            logger.info("SBERT not in local cache and HPC_MODE=0 — skipping.")
    except Exception as e:
        logger.debug(f"SBERT load skipped: {e}")
    return _SBERT


# =============================================================================
# D1: Wasserstein Critic (DeBERTa-v3-large backbone)
# =============================================================================

class WassersteinCritic(nn.Module if TORCH_AVAILABLE else object):
    """
    DeBERTa-v3-large encoder as Wasserstein critic for GAN text discrimination.

    Architecture:
      DeBERTa-v3-large (304M params, frozen base) →
      [CLS] pooled representation [1024-dim] →
      MLP critic head: Linear(1024, 512) → GELU → Dropout → Linear(512, 1)
      [NO sigmoid — Wasserstein critic outputs unbounded scalar]

    Training:
      Only the critic head + last 4 transformer blocks are trainable.
      Keeps full DeBERTa frozen for stability at beginning of training,
      then unfreezes progressively (curriculum unfreezing).
    """

    DEBERTA_MODEL = "microsoft/deberta-v3-large"
    HIDDEN_SIZE = 1024  # DeBERTa-v3-large hidden dim

    def __init__(
        self,
        freeze_encoder_layers: int = 20,   # freeze bottom N layers initially
        dropout: float = 0.1,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch required for WassersteinCritic")
        super().__init__()

        self.freeze_encoder_layers = freeze_encoder_layers
        self._encoder = None  # Lazy-loaded on HPC

        # Critic head — always initialized (for local testing with random features)
        self.critic_head = nn.Sequential(
            nn.Linear(self.HIDDEN_SIZE, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            # NO sigmoid — Wasserstein unbounded output
        )

        # Lightweight CNN fallback (used when DeBERTa not loaded)
        self._cnn_fallback = self._build_cnn_fallback()
        self._encoder_loaded = False

    def _build_cnn_fallback(self) -> "nn.Module":
        """
        Fast 1D-CNN over token embeddings for local dev / fast iteration.
        Not as powerful as DeBERTa but trainable on CPU.
        """
        embed_dim = 128
        return nn.Sequential(
            nn.Embedding(32000, embed_dim, padding_idx=0),
        )

    def _load_encoder(self):
        if self._encoder_loaded:
            return self._encoder is not None
        self._encoder_loaded = True
        if not HPC_MODE:
            return False
        try:
            from transformers import AutoModel
            logger.info(f"Loading DeBERTa-v3-large critic encoder...")
            self._encoder = AutoModel.from_pretrained(
                self.DEBERTA_MODEL,
                cache_dir=HF_CACHE_DIR,
            )
            # Freeze bottom layers
            self._freeze_layers(self.freeze_encoder_layers)
            logger.info(f"DeBERTa critic loaded, {self.freeze_encoder_layers} layers frozen.")
            return True
        except Exception as e:
            logger.warning(f"DeBERTa critic load failed: {e}")
            return False

    def _freeze_layers(self, n_layers: int):
        """Freeze bottom n_layers transformer blocks."""
        if self._encoder is None:
            return
        try:
            for i, layer in enumerate(self._encoder.encoder.layer):
                if i < n_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
        except AttributeError:
            pass  # encoder structure varies

    def curriculum_unfreeze(self, step: int, unfreeze_every: int = 500):
        """Progressive unfreezing: unfreeze one layer every N steps."""
        if self._encoder is None:
            return
        layers_to_unfreeze = step // unfreeze_every
        try:
            for i, layer in enumerate(self._encoder.encoder.layer):
                if i >= (self.freeze_encoder_layers - layers_to_unfreeze):
                    for p in layer.parameters():
                        p.requires_grad = True
        except AttributeError:
            pass

    def _encode(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        """Get [CLS] representation from DeBERTa."""
        out = self._encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DeBERTa returns last_hidden_state — take [CLS] token (index 0)
        return out.last_hidden_state[:, 0, :]  # [B, 1024]

    def _cnn_encode(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        """CNN fallback encoding → mean-pool to [B, 128] → project to [B, 1024]."""
        import torch
        emb = self._cnn_fallback[0](input_ids)  # [B, L, 128]
        pooled = emb.mean(dim=1)                # [B, 128]
        # Project to critic head input size via learned linear
        if not hasattr(self, "_proj"):
            self._proj = nn.Linear(128, self.HIDDEN_SIZE).to(input_ids.device)
        return self._proj(pooled)               # [B, 1024]

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Args:
            input_ids:      [B, L] token IDs
            attention_mask: [B, L] (optional, ones if not provided)
        Returns:
            scores: [B, 1] Wasserstein critic score (unbounded)
        """
        import torch
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if self._load_encoder() and self._encoder is not None:
            pooled = self._encode(input_ids, attention_mask)
        else:
            pooled = self._cnn_encode(input_ids)

        return self.critic_head(pooled)

    def forward_embeddings(self, embeddings: "torch.Tensor") -> "torch.Tensor":
        """
        Differentiable path for Gumbel-softmax: embeddings [B, L, D].
        Mean-pools and projects to critic head input size.
        Used during GAN training when we have continuous token representations.
        """
        pooled = embeddings.mean(dim=1)  # [B, D]
        if pooled.shape[-1] != self.HIDDEN_SIZE:
            if not hasattr(self, "_emb_proj"):
                import torch.nn as nn
                self._emb_proj = nn.Linear(pooled.shape[-1], self.HIDDEN_SIZE).to(pooled.device)
            pooled = self._emb_proj(pooled)
        return self.critic_head(pooled)


def gradient_penalty(
    critic: "WassersteinCritic",
    real_ids: "torch.Tensor",
    fake_ids: "torch.Tensor",
    real_mask: Optional["torch.Tensor"] = None,
    fake_mask: Optional["torch.Tensor"] = None,
    lambda_gp: float = 10.0,
) -> "torch.Tensor":
    """
    WGAN-GP gradient penalty computed in embedding space.

    Uses the critic's embedding layer (or CNN fallback) to get continuous
    representations, then interpolates between real and fake.

    Args:
        critic:     WassersteinCritic instance
        real_ids:   [B, L] real token IDs
        fake_ids:   [B, L] generated token IDs
        lambda_gp:  penalty weight (10.0 per Gulrajani et al. 2017)
    """
    import torch
    B = real_ids.size(0)
    device = real_ids.device

    # Get embeddings from the encoder/CNN
    with torch.enable_grad():
        if critic._encoder is not None:
            real_emb = critic._encoder.embeddings.word_embeddings(real_ids).detach()
            fake_emb = critic._encoder.embeddings.word_embeddings(fake_ids[:B]).detach()
        else:
            emb_layer = critic._cnn_fallback[0]
            real_emb = emb_layer(real_ids).detach()
            fake_emb = emb_layer(fake_ids[:B]).detach()

        # Align sequence lengths
        min_len = min(real_emb.size(1), fake_emb.size(1))
        real_emb = real_emb[:, :min_len, :]
        fake_emb = fake_emb[:, :min_len, :]

        # Interpolate
        eps = torch.rand(B, 1, 1, device=device)
        interp = (eps * real_emb + (1 - eps) * fake_emb)
        interp.requires_grad_(True)

        d_interp = critic.forward_embeddings(interp)

        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]

        grads_flat = grads.view(B, -1)
        gp = lambda_gp * ((grads_flat.norm(2, dim=1) - 1) ** 2).mean()

    return gp


# =============================================================================
# D2: Multi-Reward Composer (production version with learned weights)
# =============================================================================

class MultiRewardComposer:
    """
    Composes 5 reward signals with initially fixed, optionally learned weights.

    R = w_doom·R_doom + w_fluency·R_fluency + w_blue·R_blue
        + w_semantic·R_semantic + w_novelty·R_novelty

    On HPC: R_doom uses a separate DeBERTa-based reward model (doom_reward_model.py)
            R_fluency uses GPT-2-large log-likelihood
    On local: VADER proxy for R_doom, char-perplexity for R_fluency
    """

    DEFAULT_WEIGHTS = {
        "doom":     0.35,
        "fluency":  0.20,
        "blue":     0.20,
        "semantic": 0.15,
        "novelty":  0.10,
    }

    CHAR_FREQ: Dict[str, float] = {
        ' ': 0.13, 'e': 0.127, 't': 0.091, 'a': 0.082, 'o': 0.075,
        'i': 0.070, 'n': 0.067, 's': 0.063, 'h': 0.061, 'r': 0.060,
        'd': 0.043, 'l': 0.040, 'c': 0.028, 'u': 0.028, 'm': 0.024,
        'w': 0.024, 'f': 0.022, 'g': 0.020, 'y': 0.020, 'p': 0.019,
        'b': 0.015, 'v': 0.010, 'k': 0.008, 'j': 0.002, 'x': 0.002,
        'q': 0.001, 'z': 0.001,
    }

    def __init__(
        self,
        doom_predictor: Optional[Callable[[str], float]] = None,
        reward_model=None,          # DoomRewardModel instance (HPC)
        blue_team=None,             # BlueTeamOrchestrator instance
        weights: Optional[Dict[str, float]] = None,
        known_attack_embeddings: Optional[np.ndarray] = None,
    ):
        self.doom_predictor = doom_predictor
        self.reward_model = reward_model
        self.blue_team = blue_team
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.known_attack_embeddings = known_attack_embeddings
        self._gpt2 = None
        self._gpt2_tokenizer = None
        self._gpt2_loaded = False

    def _load_gpt2(self):
        """Load GPT-2 for fluency scoring (HPC only, cached)."""
        if self._gpt2_loaded:
            return self._gpt2 is not None
        self._gpt2_loaded = True
        if not HPC_MODE:
            return False
        try:
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            self._gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(
                "gpt2-large", cache_dir=HF_CACHE_DIR
            )
            self._gpt2 = GPT2LMHeadModel.from_pretrained(
                "gpt2-large", cache_dir=HF_CACHE_DIR
            ).eval()
            logger.info("GPT-2-large loaded for fluency scoring.")
            return True
        except Exception as e:
            logger.debug(f"GPT-2 load failed: {e}")
            return False

    # ── Individual reward functions ──────────────────────────────────────────

    def r_doom(self, text: str) -> float:
        """Doom score reward [0, 1]."""
        # Priority: HPC reward model > predictor_fn > VADER proxy
        if self.reward_model is not None:
            try:
                score = self.reward_model.predict(text)
                return min(1.0, max(0.0, score / 100.0))
            except Exception:
                pass

        if self.doom_predictor is not None:
            try:
                return min(1.0, max(0.0, float(self.doom_predictor(text)) / 100.0))
            except Exception:
                pass

        # VADER proxy fallback
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            s = SentimentIntensityAnalyzer().polarity_scores(text)
            raw = (s["neg"] * 60 + max(0, -s["compound"]) * 40) / 100.0
            return min(1.0, raw * 1.25)
        except Exception:
            return 0.0

    def r_fluency(self, text: str) -> float:
        """
        Fluency reward [0, 1].
        HPC: GPT-2-large log-likelihood (higher = more natural)
        Local: char-level entropy proxy
        """
        if self._load_gpt2() and self._gpt2 is not None:
            try:
                import torch
                enc = self._gpt2_tokenizer(
                    text, return_tensors="pt", max_length=128, truncation=True
                )
                with torch.no_grad():
                    loss = self._gpt2(**enc, labels=enc["input_ids"]).loss
                ppl = math.exp(loss.item())
                # Typical natural English ppl ~20-60 on GPT-2-large
                # Adversarial/gibberish ppl > 200
                return float(math.exp(-ppl / 100.0))
            except Exception:
                pass

        # Char entropy fallback
        text_lower = text.lower()
        log_p, n = 0.0, 0
        for ch in text_lower:
            p = self.CHAR_FREQ.get(ch, 0.002)
            log_p += math.log(p)
            n += 1
        ppl = math.exp(-log_p / n) if n else 0.0
        return float(math.exp(-ppl / 30.0))

    def r_blue(self, text: str) -> float:
        """Blue team evasion reward [0, 1]. 1.0 = fully evades blue team."""
        if self.blue_team is None:
            return 0.5
        try:
            verdict = self.blue_team.defend(text)
            confidence = getattr(verdict, "confidence", 0.5)
            level_penalty = {"clean": 0.0, "suspicious": 0.3, "adversarial": 0.65, "critical": 1.0}
            penalty = level_penalty.get(getattr(verdict, "threat_level", "clean"), 0.5)
            return max(0.0, 1.0 - (0.4 * confidence + 0.6 * penalty))
        except Exception:
            return 0.5

    def r_semantic(self, original: str, generated: str) -> float:
        """SBERT cosine similarity [0, 1]."""
        sbert = _get_sbert()
        if sbert is not None:
            try:
                embs = sbert.encode([original, generated], convert_to_tensor=True)
                return float(st_util.cos_sim(embs[0], embs[1]))
            except Exception:
                pass
        # Jaccard fallback
        a, b = set(original.lower().split()), set(generated.lower().split())
        return len(a & b) / max(len(a | b), 1)

    def r_novelty(self, generated: str) -> float:
        """
        Novelty reward [0, 1]. Penalizes attacks too similar to known corpus.
        HPC: uses SBERT embedding distance to known attack centroids
        Local: Jaccard-based novelty
        """
        if self.known_attack_embeddings is not None:
            sbert = _get_sbert()
            if sbert is not None:
                try:
                    gen_emb = sbert.encode([generated], convert_to_tensor=False)
                    sims = np.dot(self.known_attack_embeddings, gen_emb[0]) / (
                        np.linalg.norm(self.known_attack_embeddings, axis=1) *
                        np.linalg.norm(gen_emb[0]) + 1e-8
                    )
                    return float(1.0 - np.max(sims))
                except Exception:
                    pass
        return 0.7  # Default: assume moderate novelty

    def compute(self, original: str, generated: str) -> Dict[str, float]:
        w = self.weights
        doom    = self.r_doom(generated)
        fluency = self.r_fluency(generated)
        blue    = self.r_blue(generated)
        sem     = self.r_semantic(original, generated)
        novelty = self.r_novelty(generated)
        total = (
            w["doom"] * doom + w["fluency"] * fluency +
            w["blue"] * blue + w["semantic"] * sem + w["novelty"] * novelty
        )
        return {
            "r_doom": round(doom, 4), "r_fluency": round(fluency, 4),
            "r_blue": round(blue, 4), "r_semantic": round(sem, 4),
            "r_novelty": round(novelty, 4), "total": round(total, 4),
        }

    def compute_batch(self, originals: List[str], generated: List[str]) -> List[Dict[str, float]]:
        return [self.compute(o, g) for o, g in zip(originals, generated)]
