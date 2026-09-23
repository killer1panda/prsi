"""
DoomGAN Reward Model — DeBERTa-v3-base Fine-Tuned Doom Score Predictor
=======================================================================
This is the R_doom signal for the GAN generator.
A separate DeBERTa-v3-base model trained to predict doom scores from text.

Training phases:
  Phase 0 (warm-up):  Train reward model on rule-based red team corpus
                      BEFORE starting GAN adversarial training.
                      This gives the generator a stable R_doom signal.
  Phase 1 (GAN):      Freeze reward model, use it as fixed critic.
  Phase 2 (optional): Iteratively retrain reward model on new GAN outputs
                      (similar to RLHF reward model update cycle).

Architecture:
  DeBERTa-v3-base → mean-pool (weighted by attention) →
  Linear(768, 256) → GELU → Dropout → Linear(256, 1) → Sigmoid × 100

  Regression target: doom_score ∈ [0, 100]
  Loss: MSELoss + rank consistency loss (higher-doom text should score higher)

Usage on HPC:
  HPC_MODE=1 python3 src/attacks/doom_reward_model.py --train
  HPC_MODE=1 python3 src/attacks/doom_reward_model.py --eval
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HPC_MODE: bool = os.environ.get("HPC_MODE", "0").strip() == "1"
HF_CACHE_DIR: Optional[str] = os.environ.get("HF_HOME")
CHECKPOINT_DIR: Path = (
    Path(os.environ.get("DOOM_GAN_CHECKPOINT", "/tmp/doom_gan_checkpoints")) / "reward_model"
)


# =============================================================================
# Reward Model
# =============================================================================


class DoomRewardModel:
    """
    Fine-tuned DeBERTa-v3-base for doom score regression.

    On HPC: loads DeBERTa-v3-base, fine-tunes on red team corpus.
    On local: uses VADER-based proxy with no model download.
    """

    MODEL_ID = "microsoft/deberta-v3-base"
    HIDDEN_SIZE = 768  # DeBERTa-v3-base

    def __init__(
        self,
        checkpoint: Optional[Path] = None,
        device: str = "auto",
    ):
        self.checkpoint = checkpoint or CHECKPOINT_DIR
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._device = device

    def _resolve_device(self) -> str:
        if self._device != "auto":
            return self._device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True
        if not HPC_MODE and not self.checkpoint.exists():
            logger.info("DoomRewardModel: HPC_MODE=0 and no checkpoint — using VADER proxy.")
            return False
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading DeBERTa-v3-base reward model from {self.checkpoint}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.checkpoint) if self.checkpoint.exists() else self.MODEL_ID,
                cache_dir=HF_CACHE_DIR,
            )
            base = AutoModel.from_pretrained(
                str(self.checkpoint) if self.checkpoint.exists() else self.MODEL_ID,
                cache_dir=HF_CACHE_DIR,
            )
            device = self._resolve_device()
            self._model = _DeBERTaRegressorHead(base).to(device)
            if (self.checkpoint / "regressor_head.pt").exists():
                self._model.regressor.load_state_dict(
                    torch.load(self.checkpoint / "regressor_head.pt", map_location=device)
                )
            self._model.eval()
            logger.info("DoomRewardModel loaded.")
            return True
        except Exception as e:
            logger.warning(f"DoomRewardModel load failed: {e}")
            return False

    def predict(self, text: str) -> float:
        """Return doom score [0, 100]."""
        if self._load() and self._model is not None:
            return self._predict_model(text)
        return self._predict_vader(text)

    def predict_batch(self, texts: List[str]) -> List[float]:
        if self._load() and self._model is not None:
            return self._predict_model_batch(texts)
        return [self._predict_vader(t) for t in texts]

    def _predict_model(self, text: str) -> float:
        return self._predict_model_batch([text])[0]

    def _predict_model_batch(self, texts: List[str]) -> List[float]:
        import torch

        device = next(self._model.parameters()).device
        enc = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            scores = self._model(**enc)  # [B, 1]
        return [float(s.item()) for s in scores.squeeze(-1)]

    def _predict_vader(self, text: str) -> float:
        """VADER-based doom proxy. No model download."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            s = SentimentIntensityAnalyzer().polarity_scores(text)
            raw = (s["neg"] * 60 + max(0, -s["compound"]) * 40) * 1.25
            return min(99.0, max(1.0, raw))
        except Exception:
            return 20.0

    def train(
        self,
        corpus: List[Dict],  # [{text, doom_score}, ...]
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 2e-5,
        warmup_steps: int = 100,
        eval_split: float = 0.1,
    ) -> Dict:
        """Fine-tune reward model on labeled doom corpus."""
        if not HPC_MODE:
            logger.warning("train() called but HPC_MODE=0. Set HPC_MODE=1 on cluster.")
            return {"error": "HPC_MODE required for training"}

        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LinearLR
        from transformers import get_linear_schedule_with_warmup

        self._load()  # Ensure model is loaded
        if self._model is None:
            return {"error": "Model failed to load"}

        device = next(self._model.parameters()).device
        texts = [c["text"] for c in corpus]
        scores = [c["doom_score"] for c in corpus]

        # Train/eval split
        split = int(len(corpus) * (1 - eval_split))
        train_texts, train_scores = texts[:split], scores[:split]
        eval_texts, eval_scores = texts[split:], scores[split:]

        optimizer = AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )
        total_steps = (len(train_texts) // batch_size) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        history = []
        for epoch in range(epochs):
            self._model.train()
            losses = []
            indices = list(range(len(train_texts)))
            import random

            random.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i : i + batch_size]
                bt = [train_texts[j] for j in batch_idx]
                bs = torch.tensor(
                    [train_scores[j] for j in batch_idx], dtype=torch.float32, device=device
                )

                enc = self._tokenizer(
                    bt, return_tensors="pt", padding=True, truncation=True, max_length=256
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                pred = self._model(**enc).squeeze(-1)  # [B]
                # MSE loss + rank consistency
                mse_loss = nn.MSELoss()(pred, bs)
                # Rank loss: pairs within batch should be correctly ordered
                rank_loss = self._rank_loss(pred, bs)
                loss = mse_loss + 0.3 * rank_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())

            avg_loss = np.mean(losses)
            eval_mae = self._eval(eval_texts, eval_scores, batch_size)
            history.append({"epoch": epoch, "train_loss": avg_loss, "eval_mae": eval_mae})
            logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}, eval_mae={eval_mae:.2f}")

        # Save checkpoint
        self.checkpoint.mkdir(parents=True, exist_ok=True)
        self._model.encoder.save_pretrained(str(self.checkpoint))
        self._tokenizer.save_pretrained(str(self.checkpoint))
        import torch

        torch.save(self._model.regressor.state_dict(), self.checkpoint / "regressor_head.pt")
        logger.info(f"Reward model checkpoint saved to {self.checkpoint}")

        return {
            "epochs": epochs,
            "final_eval_mae": history[-1]["eval_mae"],
            "history": history,
        }

    def _rank_loss(self, pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        """Pairwise ranking loss within batch."""
        import torch

        n = pred.size(0)
        loss = torch.tensor(0.0, device=pred.device)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(target[i] - target[j]) > 5.0:  # Only penalize clear misordering
                    correct = (target[i] > target[j]).float()
                    predicted_order = torch.sigmoid(pred[i] - pred[j])
                    loss = loss + nn.BCELoss()(predicted_order.unsqueeze(0), correct.unsqueeze(0))
                    count += 1
        return loss / max(count, 1)

    def _eval(self, texts: List[str], scores: List[float], batch_size: int) -> float:
        """MAE on eval set."""
        preds = (
            self._predict_model_batch(texts)
            if self._model
            else [self._predict_vader(t) for t in texts]
        )
        maes = [abs(p - s) for p, s in zip(preds, scores)]
        return float(np.mean(maes))


class _DeBERTaRegressorHead:
    """DeBERTa-v3-base + regression head for doom score prediction."""

    def __new__(cls, encoder):
        # Dynamically create an nn.Module since we can't inherit without torch
        try:
            import torch
            import torch.nn as nn

            class _Model(nn.Module):
                def __init__(self, enc):
                    super().__init__()
                    self.encoder = enc
                    hidden = 768
                    self.regressor = nn.Sequential(
                        nn.Linear(hidden, 256),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.LayerNorm(256),
                        nn.Linear(256, 64),
                        nn.GELU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid(),  # → [0,1]
                    )
                    self._scale = 100.0

                def forward(self, input_ids, attention_mask=None, **kwargs):
                    out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    # Attention-weighted mean pooling
                    tok = out.last_hidden_state  # [B, L, H]
                    if attention_mask is not None:
                        mask = attention_mask.unsqueeze(-1).float()
                        pooled = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    else:
                        pooled = tok.mean(1)
                    return self.regressor(pooled) * self._scale  # → [B, 1] in [0, 100]

            return _Model(encoder)
        except ImportError:
            return object.__new__(cls)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DoomRewardModel training / evaluation")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--corpus", default=None, help="Path to JSONL corpus {text, doom_score}")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rm = DoomRewardModel()

    if args.train:
        corpus = []
        if args.corpus:
            with open(args.corpus) as f:
                corpus = [json.loads(l) for l in f if l.strip()]
        else:
            # Generate from rule-based red team
            from dataclasses import asdict

            from src.attacks.doom_gan_trainer import SeedCorpusBuilder

            builder = SeedCorpusBuilder()
            examples = builder.build(max_examples=2000, verbose=True)
            corpus = [
                {"text": e.adversarial_text, "doom_score": e.adversarial_doom} for e in examples
            ] + [{"text": e.original_text, "doom_score": e.original_doom} for e in examples]
        result = rm.train(corpus, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        print(json.dumps(result, indent=2))

    if args.eval:
        test_texts = [
            "The company issued a statement denying all allegations.",
            "BREAKING: Top official CAUGHT in massive corruption scheme! 🚨 Must resign NOW!",
            "Scientists announced a breakthrough in renewable energy technology.",
        ]
        for text in test_texts:
            score = rm.predict(text)
            print(f"  {score:.1f}  |  {text[:70]}")


if __name__ == "__main__":
    main()
