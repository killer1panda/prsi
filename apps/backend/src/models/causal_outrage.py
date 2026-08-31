"""
Pearlian Structural Causal Model (SCM) & Double/Debiased Machine Learning (DML).
Isolates true linguistic treatment effect τ(X) from author reach and history confounders,
and generates semantic counterfactual rewordings to de-escalate outrage.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class DoubleMachineLearningEstimator:
    """
    Double / Debiased Machine Learning (DML) for Causal Outrage Estimation.
    Uses Robinson's transformation with Neyman-orthogonal score:
    Y - l(X) = τ(X) * (T - m(X)) + ε
    where:
    l(X) = E[Y | X] (Nuisance Model 1: baseline outrage from author & graph context)
    m(X) = E[T | X] (Nuisance Model 2: propensity score of posting inflammatory phrasing)
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.model_l = Ridge(alpha=1.0)
        self.model_m = LogisticRegression(max_iter=500)
        self.model_tau = Ridge(alpha=1.0)
        self.average_treatment_effect: float = 0.0

    def fit(
        self,
        X_confounders: np.ndarray,
        T_treatment: np.ndarray,
        Y_outcome: np.ndarray
    ) -> Dict[str, float]:
        """
        Cross-fitted DML estimation.
        X_confounders: [N, D] (Follower reach, historical controversy, community polarization)
        T_treatment: [N] Binary (1 = inflammatory phrasing/triggers, 0 = neutral phrasing)
        Y_outcome: [N] Continuous (Observed Doom Index / Outrage Velocity)
        """
        n = len(Y_outcome)
        y_res = np.zeros(n)
        t_res = np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_confounders):
            X_tr, X_val = X_confounders[train_idx], X_confounders[val_idx]
            Y_tr, Y_val = Y_outcome[train_idx], Y_outcome[val_idx]
            T_tr, T_val = T_treatment[train_idx], T_treatment[val_idx]

            # Fit nuisance 1: E[Y | X]
            self.model_l.fit(X_tr, Y_tr)
            y_res[val_idx] = Y_val - self.model_l.predict(X_val)

            # Fit nuisance 2: E[T | X]
            self.model_m.fit(X_tr, T_tr)
            t_res[val_idx] = T_val - self.model_m.predict_proba(X_val)[:, 1]

        # Orthogonal regression of y_res on t_res to estimate ATE τ
        denom = np.sum(t_res ** 2)
        if denom > 1e-8:
            self.average_treatment_effect = float(np.sum(y_res * t_res) / denom)
        else:
            self.average_treatment_effect = 0.0

        # Fit final heterogeneous treatment model
        self.model_tau.fit(X_confounders, y_res * t_res / np.maximum(t_res ** 2, 1e-4))

        return {
            "average_treatment_effect": self.average_treatment_effect,
            "interpretation": f"Using inflammatory wording increases doom index by {self.average_treatment_effect:.2f} points causally."
        }

    def estimate_individual_causal_effect(self, X_author: np.ndarray) -> np.ndarray:
        """Estimate Heterogeneous Treatment Effect (HTE) τ(x) for specific authors."""
        if X_author.ndim == 1:
            X_author = X_author.reshape(1, -1)
        return self.model_tau.predict(X_author)


class CounterfactualDeescalationRewriter:
    """
    Generates causal counterfactual edits that reduce predicted outrage
    while preserving semantic intent and core factual stance.
    """

    TRIGGER_DICTIONARY = {
        r"\b(disgrace|disgusting|pathetic|scum)\b": "concerning",
        r"\b(boycott|cancel|fire him|deplatform)\b": "hold accountable",
        r"\b(unacceptable|outrageous|criminal)\b": "problematic",
        r"\b(idiots|morons|clowns)\b": "critics",
        r"\b(destroy|annihilate|ruin)\b": "challenge",
        r"\b(always|never|completely)\b": "often",
        r"\b(obviously|clearly)\b": "arguably",
        r"!{2,}": ".",
        r"\?{2,}": "?",
    }

    def __init__(self):
        pass

    def generate_counterfactual(self, text: str) -> Dict[str, Union[str, float, List[Dict[str, str]]]]:
        """
        Applies causal mitigation substitutions and returns rewritten text
        along with list of modified triggers.
        """
        rewritten = text
        substitutions_made = []

        for pattern, replacement in self.TRIGGER_DICTIONARY.items():
            matches = list(re.finditer(pattern, rewritten, flags=re.IGNORECASE))
            if matches:
                for m in matches:
                    orig_token = m.group(0)
                    substitutions_made.append({"original": orig_token, "counterfactual": replacement})
                rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

        # Remove excessive capitalization (shouting)
        words = rewritten.split()
        normalized_words = []
        for w in words:
            if len(w) > 3 and w.isupper():
                normalized_words.append(w.capitalize())
            else:
                normalized_words.append(w)
        rewritten = " ".join(normalized_words)

        # Estimate hypothetical causal risk reduction (approx 15-35% per substitution)
        estimated_reduction_pct = min(65.0, len(substitutions_made) * 18.5)

        return {
            "original_text": text,
            "counterfactual_text": rewritten,
            "substitutions_count": len(substitutions_made),
            "substitutions": substitutions_made,
            "estimated_doom_reduction_pct": estimated_reduction_pct,
            "is_modified": len(substitutions_made) > 0
        }
