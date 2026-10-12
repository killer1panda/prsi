"""
Pearlian Structural Causal Model (SCM) + Double ML (DML) + Causal-DPO Rewriter.

Implements a lightweight linear Gaussian DAG-based SCM for interventional and
counterfactual reasoning over outrage features, plus a rule-based DPO rewriter
that de-escalates high-outrage phrasing to reduce cancellation risk.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from doubleml import DoubleMLPLR
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    logger.info("econml/doubleml not installed; using lightweight SCM fallback")


class PearlianSCM:
    """
    Pearlian Structural Causal Model for outrage prediction.

    Encodes a linear Gaussian DAG where nodes are named feature variables
    and edges carry linear coefficients. Supports Pearl's do-calculus
    (interventions) and counterfactual queries via abduction-action-prediction.
    """

    def __init__(self):
        # DAG: variable -> {parent: coefficient}
        # doom_score = f(toxicity, outrage_phrases, emoji_valence, network_centrality)
        self.dag: Dict[str, Dict[str, float]] = {
            "outrage_phrase_density": {},
            "emoji_outrage_score": {},
            "toxicity_score": {"outrage_phrase_density": 0.65, "emoji_outrage_score": 0.45},
            "network_amplification": {},
            "doom_score": {
                "toxicity_score": 0.55,
                "outrage_phrase_density": 0.30,
                "emoji_outrage_score": 0.25,
                "network_amplification": 0.15,
            },
        }
        # Noise standard deviations per node
        self.noise_std: Dict[str, float] = {
            "outrage_phrase_density": 0.05,
            "emoji_outrage_score": 0.05,
            "toxicity_score": 0.08,
            "network_amplification": 0.10,
            "doom_score": 0.06,
        }
        self._obs: Dict[str, float] = {}

    def _compute_node(self, var: str, obs: Dict[str, float]) -> float:
        """Compute node value from parent values (structural equation)."""
        parents = self.dag.get(var, {})
        value = sum(coeff * obs.get(parent, 0.0) for parent, coeff in parents.items())
        return float(np.clip(value, 0.0, 1.0))

    def observe(self, observations: Dict[str, float]) -> None:
        """Record an observed configuration of all variables."""
        self._obs = dict(observations)

    def do_intervention(self, var: str, value: float) -> Dict[str, float]:
        """
        Pearl's do(X = value) operator.
        Removes all edges into `var`, fixes it at `value`,
        then propagates downstream effects through the DAG.

        Returns updated values for all nodes downstream of the intervention.
        """
        # Start from observed values, override the intervened variable
        result = dict(self._obs)
        result[var] = float(np.clip(value, 0.0, 1.0))

        # Topological traversal order for the DAG
        topo_order = [
            "outrage_phrase_density",
            "emoji_outrage_score",
            "network_amplification",
            "toxicity_score",
            "doom_score",
        ]

        for node in topo_order:
            if node == var:
                # Intervened: fixed, skip recomputation
                continue
            # Only recompute nodes that have intervened var as a (grand)parent
            parents = self.dag.get(node, {})
            if any(p == var or p in result for p in parents):
                result[node] = self._compute_node(node, result)

        logger.debug(f"do({var}={value:.3f}) -> doom_score={result.get('doom_score', 0):.3f}")
        return result

    def counterfactual(self, obs: Dict[str, float], intervention: Dict[str, float]) -> Dict[str, float]:
        """
        Pearl's 3-step counterfactual algorithm:
        1. Abduction: infer noise residuals from observed values
        2. Action: apply do-intervention
        3. Prediction: propagate with fixed noise residuals

        Returns the counterfactual values for all variables.
        """
        self.observe(obs)

        # Step 1: Abduction — compute residuals (obs - structural prediction)
        residuals: Dict[str, float] = {}
        for var in self.dag:
            predicted = self._compute_node(var, obs)
            residuals[var] = obs.get(var, 0.0) - predicted

        # Step 2: Action — apply intervention
        cf = self.do_intervention(
            list(intervention.keys())[0], list(intervention.values())[0]
        )

        # Step 3: Prediction — add abduced residuals back
        for var in self.dag:
            if var not in intervention:
                cf[var] = float(np.clip(cf.get(var, 0.0) + residuals.get(var, 0.0), 0.0, 1.0))

        return cf


class CausalDMLEstimator:
    """Double Machine Learning estimator for heterogeneous treatment effects."""

    def __init__(self, n_estimators: int = 200):
        self.n_estimators = n_estimators
        self._model = None
        if ECONML_AVAILABLE:
            try:
                self._model = CausalForestDML(n_estimators=self.n_estimators, random_state=42)
            except Exception as e:
                logger.warning(f"CausalForestDML init failed: {e}")

    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray, W: Optional[np.ndarray] = None) -> None:
        """
        Fit the DML estimator.
        Y: outcome (doom_score), T: treatment (e.g. emoji_outrage_score),
        X: heterogeneity features, W: nuisance controls.
        """
        if self._model is not None:
            self._model.fit(Y, T, X=X, W=W)
            logger.info("CausalForestDML fitted successfully")
        else:
            # Lightweight OLS fallback: fit linear treatment effect
            from numpy.linalg import lstsq
            n = len(Y)
            features = np.column_stack([T.reshape(-1, 1), X]) if X is not None else T.reshape(-1, 1)
            coefs, _, _, _ = lstsq(np.hstack([features, np.ones((n, 1))]), Y, rcond=None)
            self._ols_coefs = coefs
            logger.info("OLS fallback DML estimator fitted")

    def effect(self, X: np.ndarray) -> np.ndarray:
        """Estimate heterogeneous treatment effect for each unit in X."""
        if self._model is not None:
            return self._model.effect(X)
        if hasattr(self, "_ols_coefs"):
            # Return constant average treatment effect (slope of T)
            return np.full(len(X), self._ols_coefs[0])
        return np.zeros(len(X))


class CausalDPORewriter:
    """
    Direct Preference Optimization (DPO) and SCM-Guided Counterfactual Text Rewriter.
    Uses candidate generation, Pareto-optimal ranking, and causal interventional
    constraints to de-escalate text without destroying semantic meaning.
    """

    def __init__(self, beta: float = 0.1, similarity_threshold: float = 0.65):
        self.beta = beta
        self.similarity_threshold = similarity_threshold
        self.scm = PearlianSCM()

        # Transformation grammar
        self.deescalation_lexicon = {
            "fraud": ["alleged irregularity", "disputed transaction", "matter under review"],
            "corrupt": ["facing accountability questions", "under governance review", "disputed"],
            "scammer": ["controversial party", "disputed operator"],
            "liar": ["source of conflicting statements", "contested spokesperson"],
            "resign": ["step back", "reconsider their role", "transition"],
            "boycott": ["reconsider engagement", "exercise consumer choice"],
            "disgusting": ["concerning", "unacceptable to critics", "highly controversial"],
            "predator": ["individual facing serious allegations", "accused party"],
        }

        # Outrage emojis to remove or replace
        self.emoji_replacements = {
            "🤡": "🤔",
            "💀": "⚠️",
            "🚨": "📢",
            "😡": "😤",
            "🔥": "💬",
            "🗑️": "📝",
            "🤮": "😞",
            "😤": "🙁",
        }

    def _generate_candidate_pool(self, text: str) -> List[str]:
        candidates = [text]
        words = text.split()

        # Variant 1: Lexical substitution + emoji de-escalation
        v1_words = []
        for w in words:
            w_clean = re.sub(r"[^\w\s]", "", w.lower())
            if w_clean in self.deescalation_lexicon:
                sub = self.deescalation_lexicon[w_clean][0]
                v1_words.append(sub)
            else:
                v1_words.append(w)
        v1_text = " ".join(v1_words)
        for outrage_emoji, calm_emoji in self.emoji_replacements.items():
            v1_text = v1_text.replace(outrage_emoji, calm_emoji)
        candidates.append(v1_text)

        # Variant 2: Objective attribution framing
        candidates.append(f"Reports indicate that {candidates[-1].lower()}")

        # Variant 3: Epistemic softening
        softened = text
        for pat in [r"\balways\b", r"\bnever\b", r"\beveryone\b"]:
            softened = re.sub(pat, "frequently", softened, flags=re.IGNORECASE)
        candidates.append(softened)

        return list(set(candidates))

    def _compute_dpo_reward(self, original: str, candidate: str) -> float:
        """
        Implicit Bradley-Terry reward scoring:
        R(y) = - Causal_Doom_Risk(y) + λ * Semantic_Overlap(x, y)
        """
        try:
            from src.features.sentiment import analyze_text_sentiment
            from src.features.toxicity import analyze_text_toxicity

            s = analyze_text_sentiment(candidate) or {}
            t = analyze_text_toxicity(candidate) or {}

            obs = {
                "outrage_phrase_density": max(0.0, s.get("sentiment_negative", 0.0)),
                "emoji_outrage_score": float(len(re.findall(r"[🚨😡🔥💀]", candidate)) > 0),
                "toxicity_score": float(t.get("toxicity_score", 0.0)),
                "network_amplification": 0.5,
                "doom_score": 0.5,
            }
        except Exception:
            obs = {
                "outrage_phrase_density": 0.3,
                "emoji_outrage_score": 0.0,
                "toxicity_score": 0.2,
                "network_amplification": 0.5,
                "doom_score": 0.4,
            }

        # Evaluate through Pearlian SCM
        intervened = self.scm.do_intervention("outrage_phrase_density", obs["outrage_phrase_density"])
        predicted_doom = intervened.get("doom_score", 0.5)

        # Word Jaccard semantic overlap
        w_orig = set(original.lower().split())
        w_cand = set(candidate.lower().split())
        jaccard = len(w_orig & w_cand) / max(len(w_orig | w_cand), 1)

        # DPO score: minimize doom risk while retaining semantic intent
        reward = (1.0 - predicted_doom) * 1.5 + jaccard * 1.0
        return reward

    def rewrite(self, text: str) -> str:
        """Select Pareto-optimal de-escalation candidate."""
        if not text or len(text.strip()) == 0:
            return text

        candidates = self._generate_candidate_pool(text)
        scored = [(c, self._compute_dpo_reward(text, c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def generate_variants(self, text: str, n: int = 2) -> List[str]:
        candidates = self._generate_candidate_pool(text)
        scored = [(c, self._compute_dpo_reward(text, c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:n]]
