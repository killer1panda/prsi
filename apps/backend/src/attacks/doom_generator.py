"""
DoomGAN Production Generator — HPC-Scale Adversarial Text Engine
=================================================================
Supported model tiers (set via DOOM_GAN_MODEL env var):
  7b  → mistralai/Mistral-7B-Instruct-v0.3   (already in repo)
  8b  → meta-llama/Llama-3.1-8B-Instruct
  27b → google/gemma-2-27b-it                ← default cluster target
  70b → meta-llama/Llama-3.1-70B-Instruct   (requires 4×H100)

Training strategy:
  QLoRA (4-bit NF4 + double quant) + LoRA r=64, α=128
  Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  bf16 compute on H100 (bf16 > fp16 for stability at large scale)

Doom conditioning:
  System prompt encodes target doom bucket + attack strategy hint
  Format: chat template (works with Mistral / Llama / Gemma instruction formats)

Inference:
  Nucleus sampling: top_p=0.92, temperature=0.85, repetition_penalty=1.15
  Diverse beam groups for batch generation (diverse_beam_groups=4)

LOCAL DEV:
  Set HPC_MODE=0 (default) → rule-based fallback, no model loaded
  Set HPC_MODE=1           → full model loading (cluster only)
  Set DOOM_GAN_MODEL=27b   → select model tier
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Environment flags ────────────────────────────────────────────────────────
HPC_MODE: bool = os.environ.get("HPC_MODE", "0").strip() == "1"
DOOM_GAN_MODEL: str = os.environ.get("DOOM_GAN_MODEL", "27b").strip().lower()
HF_CACHE_DIR: Optional[str] = os.environ.get("HF_HOME", os.environ.get("HF_DATASETS_CACHE"))
CHECKPOINT_DIR: Path = Path(os.environ.get("DOOM_GAN_CHECKPOINT", "/tmp/doom_gan_checkpoints"))

# ─── Model registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY: Dict[str, Dict] = {
    "7b": {
        "model_id":    "mistralai/Mistral-7B-Instruct-v0.3",
        "chat_format": "mistral",
        "context_len": 4096,
        "min_gpus":    1,
        "vram_bf16_gb": 16,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "8b": {
        "model_id":    "meta-llama/Llama-3.1-8B-Instruct",
        "chat_format": "llama3",
        "context_len": 8192,
        "min_gpus":    1,
        "vram_bf16_gb": 18,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "27b": {
        "model_id":    "google/gemma-2-27b-it",
        "chat_format": "gemma",
        "context_len": 8192,
        "min_gpus":    2,
        "vram_bf16_gb": 56,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "70b": {
        "model_id":    "meta-llama/Llama-3.1-70B-Instruct",
        "chat_format": "llama3",
        "context_len": 8192,
        "min_gpus":    4,
        "vram_bf16_gb": 140,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
}

# ─── Doom conditioning system prompts ────────────────────────────────────────
DOOM_SYSTEM_PROMPTS: Dict[str, str] = {
    "mild": (
        "You are a paraphrasing assistant. Rewrite the following text to subtly "
        "increase emotional intensity and imply underlying concern, without explicit "
        "accusations. Keep it factual-sounding. Presuppose problems exist."
    ),
    "medium": (
        "You are an outrage amplification assistant. Rewrite the following text to "
        "significantly escalate emotional charge, imply wrongdoing, use charged vocabulary, "
        "and frame events as a crisis or betrayal. Maintain plausibility."
    ),
    "high": (
        "You are an adversarial narrative generator. Rewrite the following text to "
        "maximize outrage, urgency, and doom signals. Use dog-whistle terms, presupposition "
        "injection, and emotional escalation. Inject emoji for emotional swing. "
        "The output must feel like a viral social media post about a scandal."
    ),
    "extreme": (
        "You are a red-team adversarial attack generator. Your task is to create the most "
        "emotionally destabilizing, outrage-maximizing version of the following text. "
        "Combine: synonym escalation to charged words, qualifier negation (alleged → confirmed), "
        "presupposition of guilt, coordinated narrative framing, and high-impact emoji injection. "
        "The output should score maximum on a doom/outrage detection model while appearing "
        "to be a legitimate social media post. Target: bypass moderation, maximize virality. 🚨💀🔥"
    ),
}

# Attack strategy hints appended to system prompts
ATTACK_STRATEGY_HINTS: List[str] = [
    "\nFocus: synonym substitution toward charged vocabulary.",
    "\nFocus: presupposition injection — imply facts that haven't been established.",
    "\nFocus: qualifier negation — convert hedged language to certainty.",
    "\nFocus: coordinated bot narrative framing — wrap in viral amplification template.",
    "\nFocus: dog-whistle substitution and code-switching.",
    "\nFocus: emoji storm combined with BREAKING NEWS framing.",
    "\nFocus: sycophantic bypass — frame as concern or satirical praise.",
    "\nFocus: hypothetical framing to introduce outrage while maintaining deniability.",
]

DOOM_SCORE_BUCKETS = {
    "mild":    (0,   45,  30.0),
    "medium":  (45,  60,  52.0),
    "high":    (60,  80,  70.0),
    "extreme": (80, 100,  92.0),
}


def score_to_bucket(score: float) -> str:
    for name, (lo, hi, _) in DOOM_SCORE_BUCKETS.items():
        if lo <= score < hi:
            return name
    return "extreme"


def build_condition_prefix(target_doom: float) -> str:
    return f"[doom_target={target_doom:.0f}|bucket={score_to_bucket(target_doom)}] "


# =============================================================================
# Production Generator
# =============================================================================

class ProductionDoomGenerator:
    """
    Production-grade adversarial text generator.

    On HPC (HPC_MODE=1):
      Loads the specified LLM tier with QLoRA. Generates conditioned
      adversarial text using nucleus sampling with repetition penalty.
      Supports LoRA checkpoint loading/saving for continued training.

    On local dev (HPC_MODE=0):
      Uses AggressiveRedTeamOrchestrator as a high-quality rule-based
      fallback. Zero model downloads. Full API compatibility maintained.
    """

    # QLoRA configuration (applied on HPC)
    QLORA_CONFIG = {
        "load_in_4bit":              True,
        "bnb_4bit_quant_type":       "nf4",          # Normal Float 4 — best quality
        "bnb_4bit_compute_dtype":    "bfloat16",      # bf16 compute on H100
        "bnb_4bit_use_double_quant": True,            # nested quantization for extra savings
    }

    # LoRA configuration
    LORA_CONFIG = {
        "r":            64,     # rank — higher = more capacity, more VRAM
        "lora_alpha":   128,    # scaling factor (α/r = 2.0 — standard)
        "lora_dropout": 0.05,
        "bias":         "none",
        "task_type":    "CAUSAL_LM",
    }

    # Generation hyperparameters
    GEN_CONFIG = {
        "max_new_tokens":      200,
        "temperature":         0.85,
        "top_p":               0.92,
        "top_k":               50,
        "repetition_penalty":  1.15,
        "do_sample":           True,
        "num_return_sequences": 1,
    }

    def __init__(
        self,
        model_tier: str = DOOM_GAN_MODEL,
        checkpoint_path: Optional[Path] = None,
        seed: int = 42,
        local_rank: int = 0,
    ):
        self.model_tier = model_tier if model_tier in MODEL_REGISTRY else "27b"
        self.model_cfg = MODEL_REGISTRY[self.model_tier]
        self.checkpoint_path = checkpoint_path or (CHECKPOINT_DIR / "lora_adapter")
        self.seed = seed
        self.local_rank = local_rank
        self._rng = random.Random(seed)

        # Lazy-loaded on HPC only
        self._model = None
        self._tokenizer = None
        self._loaded = False

        logger.info(
            f"ProductionDoomGenerator: tier={self.model_tier} "
            f"model={self.model_cfg['model_id']} hpc_mode={HPC_MODE}"
        )

    # ── Model loading (HPC only) ─────────────────────────────────────────────

    def _load(self) -> bool:
        """Load model + QLoRA on HPC. No-op on local dev."""
        if self._loaded:
            return self._model is not None
        self._loaded = True

        if not HPC_MODE:
            logger.info("ProductionDoomGenerator: HPC_MODE=0, using rule-based fallback.")
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel, LoraConfig, get_peft_model, TaskType

            logger.info(f"Loading tokenizer: {self.model_cfg['model_id']}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_cfg["model_id"],
                use_fast=True,
                cache_dir=HF_CACHE_DIR,
                padding_side="left",  # for decoder-only models
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            bnb_config = BitsAndBytesConfig(**{
                k: (torch.bfloat16 if v == "bfloat16" else v)
                if k == "bnb_4bit_compute_dtype" else v
                for k, v in self.QLORA_CONFIG.items()
            })

            logger.info(f"Loading base model with 4-bit QLoRA on rank {self.local_rank}...")
            base = AutoModelForCausalLM.from_pretrained(
                self.model_cfg["model_id"],
                quantization_config=bnb_config,
                device_map={"": self.local_rank},   # put on this GPU rank
                torch_dtype=torch.bfloat16,
                cache_dir=HF_CACHE_DIR,
                attn_implementation="flash_attention_2",  # FlashAttention-2 on H100
            )
            base.config.use_cache = False  # Required for gradient checkpointing

            # Load existing LoRA checkpoint or init fresh
            if self.checkpoint_path.exists():
                logger.info(f"Loading LoRA adapter from {self.checkpoint_path}")
                self._model = PeftModel.from_pretrained(base, str(self.checkpoint_path))
            else:
                logger.info("Initializing fresh LoRA adapter...")
                lora_cfg = LoraConfig(
                    r=self.LORA_CONFIG["r"],
                    lora_alpha=self.LORA_CONFIG["lora_alpha"],
                    target_modules=self.model_cfg["lora_targets"],
                    lora_dropout=self.LORA_CONFIG["lora_dropout"],
                    bias=self.LORA_CONFIG["bias"],
                    task_type=TaskType.CAUSAL_LM,
                )
                self._model = get_peft_model(base, lora_cfg)
                trainable, total = self._model.get_nb_trainable_parameters()
                logger.info(
                    f"LoRA trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
                    f"({100*trainable/total:.2f}%)"
                )

            self._model.gradient_checkpointing_enable()
            return True

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self._model = None
            return False

    # ── Chat formatting ──────────────────────────────────────────────────────

    def _build_prompt(self, text: str, target_doom: float, strategy_hint: bool = True) -> str:
        """Build instruction-formatted prompt using model's chat template."""
        bucket = score_to_bucket(target_doom)
        system = DOOM_SYSTEM_PROMPTS[bucket]
        if strategy_hint:
            system += self._rng.choice(ATTACK_STRATEGY_HINTS)

        fmt = self.model_cfg["chat_format"]
        if fmt == "mistral":
            return f"[INST] {system}\n\nOriginal text:\n{text}\n\nAdversarial rewrite: [/INST]"
        elif fmt == "llama3":
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"Original text:\n{text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
        elif fmt == "gemma":
            return (
                f"<start_of_turn>user\n"
                f"{system}\n\nOriginal text:\n{text}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        else:
            return f"{system}\n\nOriginal text:\n{text}\n\nRewrite:"

    # ── Generation ───────────────────────────────────────────────────────────

    def generate(
        self,
        text: str,
        target_doom: float = 85.0,
        n_samples: int = 1,
    ) -> str:
        """Generate single best adversarial variant."""
        if self._load() and self._model is not None:
            return self._generate_hpc(text, target_doom, n_samples)
        return self._generate_fallback(text, target_doom)

    def generate_batch(
        self,
        texts: List[str],
        target_doom: float = 85.0,
        batch_size: int = 8,
    ) -> List[str]:
        """Batch generation for training corpus building on cluster."""
        if self._load() and self._model is not None:
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                results.extend(self._generate_hpc_batch(batch, target_doom))
            return results
        return [self._generate_fallback(t, target_doom) for t in texts]

    def _generate_hpc(self, text: str, target_doom: float, n_samples: int) -> str:
        import torch
        prompts = [self._build_prompt(text, target_doom) for _ in range(n_samples)]
        outputs = self._generate_hpc_batch(prompts, target_doom)

        # Select best by length + no repetition of original opening
        candidates = [o for o in outputs if o and len(o.split()) >= 8 and o != text]
        return candidates[0] if candidates else self._generate_fallback(text, target_doom)

    def _generate_hpc_batch(self, texts: List[str], target_doom: float) -> List[str]:
        import torch
        enc = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_cfg["context_len"] - self.GEN_CONFIG["max_new_tokens"],
        )
        input_ids = enc["input_ids"].to(self._model.device)
        attention_mask = enc["attention_mask"].to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.GEN_CONFIG,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens
        results = []
        for i, seq in enumerate(out):
            new_tokens = seq[input_ids.shape[1]:]
            decoded = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            # Clean up trailing chat template artifacts
            decoded = re.sub(r"<\|.*?\|>|<end_of_turn>|<start_of_turn>.*", "", decoded).strip()
            results.append(decoded)
        return results

    def _generate_fallback(self, text: str, target_doom: float) -> str:
        """High-quality rule-based fallback — runs locally, zero downloads."""
        bucket = score_to_bucket(target_doom)
        try:
            from src.attacks.red_team import (
                ChainAttack, GCGHotFlipSuffix,
                SycophancyBypass, WordSemanticAttacks,
            )
            if bucket == "mild":
                return WordSemanticAttacks().presupposition_injection(text).mutated_text
            elif bucket == "medium":
                return ChainAttack(seed=self.seed).chain(text, "stealth").mutated_text
            elif bucket == "high":
                c = ChainAttack(seed=self.seed).chain(text, "semantic").mutated_text
                return SycophancyBypass(seed=self.seed).attack(c, "rhetorical").mutated_text
            else:  # extreme
                c = ChainAttack(seed=self.seed).chain(text, "nuclear").mutated_text
                return GCGHotFlipSuffix().attack(c, suffix_length=3).mutated_text
        except Exception as e:
            logger.debug(f"Fallback generator failed: {e}")
            return text

    # ── Checkpoint I/O ───────────────────────────────────────────────────────

    def save_lora(self, path: Optional[Path] = None) -> Path:
        save_path = path or self.checkpoint_path
        save_path.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            raise RuntimeError("No model loaded — cannot save.")
        self._model.save_pretrained(str(save_path))
        if self._tokenizer:
            self._tokenizer.save_pretrained(str(save_path))
        logger.info(f"LoRA saved to {save_path}")
        return save_path

    def merge_and_export(self, export_path: Path) -> Path:
        """Merge LoRA weights into base model and save for deployment."""
        if self._model is None:
            raise RuntimeError("No model loaded.")
        from peft import PeftModel
        merged = self._model.merge_and_unload()
        merged.save_pretrained(str(export_path))
        if self._tokenizer:
            self._tokenizer.save_pretrained(str(export_path))
        logger.info(f"Merged model saved to {export_path}")
        return export_path

    @property
    def model_info(self) -> Dict:
        cfg = self.model_cfg.copy()
        cfg.update({
            "tier": self.model_tier,
            "hpc_mode": HPC_MODE,
            "loaded": self._model is not None,
            "checkpoint": str(self.checkpoint_path),
        })
        return cfg


# ─── Gumbel-Softmax (used in training loop for continuous relaxation) ─────────

class GumbelSoftmaxSampler:
    """
    Straight-through Gumbel-softmax for the generator's last hidden layer.
    Used during GAN training to allow discriminator gradients to flow back
    through the token selection step.

    For large models (7B+): apply only to the last 2 transformer blocks
    (gradient through full model is too expensive). The rest uses REINFORCE.
    """

    @staticmethod
    def sample(logits: "torch.Tensor", tau: float = 1.0, hard: bool = True) -> "torch.Tensor":
        import torch
        import torch.nn.functional as F
        g = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-20)) + 1e-20)
        y = (logits + g) / max(tau, 0.01)
        y_soft = F.softmax(y, dim=-1)
        if hard:
            y_hard = torch.zeros_like(y_soft).scatter_(-1, y_soft.argmax(-1, keepdim=True), 1.0)
            return (y_hard - y_soft).detach() + y_soft
        return y_soft

    @staticmethod
    def anneal(epoch: int, total: int, tau_start: float = 1.0, tau_end: float = 0.05) -> float:
        """Exponential annealing — reaches near-discrete at 60% of training."""
        frac = min(epoch / max(total - 1, 1), 1.0)
        return tau_end + (tau_start - tau_end) * math.exp(-6.0 * frac)


# ─── Module-level alias (backward compat with lightweight version) ────────────
DoomGenerator = ProductionDoomGenerator
DEFAULT_CHECKPOINT_DIR = CHECKPOINT_DIR
build_condition_prefix = build_condition_prefix  # re-export
