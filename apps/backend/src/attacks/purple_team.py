"""
PURPLE TEAM ENGINE — PRSI Doom-Index Continuous Red/Blue Feedback Loop
=======================================================================
Purple Team = Red Team attacks + Blue Team defenses running together
in a continuous adversarial co-evolution cycle.

Purple Team Capabilities:
  P1. Continuous Attack Surface Profiling
      — runs all red-team attacks, ranks by effectiveness
      — tracks which attack types consistently bypass blue team
  P2. Defense Effectiveness Dashboard
      — measures detection rate, false positive rate per attack type
      — tracks defense degradation over time
  P3. Adaptive Threshold Tuning
      — automatically tightens blue-team thresholds when attacks succeed
      — loosens thresholds when FPR exceeds acceptable level
  P4. Adversarial Training Signal Generation
      — exports successful (bypassing) attack examples for model hardening
      — formats as augmented training dataset with labels
  P5. Full Engagement Report
      — generates detailed JSON report: attack effectiveness matrix,
        defense coverage gaps, recommended remediation actions
  P6. Real-Time Monitoring Integration
      — hooks into /events SSE stream, scores incoming posts in real-time,
        runs blue team defense simultaneously, flags coordinated campaigns
  P7. MITRE ATT&CK for NLP Mapping
      — maps each attack to MITRE-style technique IDs for auditability

Usage:
    from src.attacks.purple_team import PurpleTeamOrchestrator
    purple = PurpleTeamOrchestrator(predictor_fn=doom_predictor)
    report = purple.full_engagement(text="This CEO must resign NOW!!!")
    print(report.summary())
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.attacks.red_team import (
    RedTeamOrchestrator,
    RedTeamResult,
)
from src.attacks.blue_team import BlueTeamOrchestrator, BlueTeamVerdict

logger = logging.getLogger(__name__)

# ─── MITRE ATT&CK for NLP technique mapping ───────────────────────────────────

ATTACK_TECHNIQUE_MAP: Dict[str, Dict[str, str]] = {
    "HomoglyphSubstitution":    {"id": "T-NLP-001", "tactic": "Defense Evasion", "subtactic": "Obfuscation"},
    "ZeroWidthInjection":       {"id": "T-NLP-002", "tactic": "Defense Evasion", "subtactic": "Steganography"},
    "LeetSpeak":                {"id": "T-NLP-003", "tactic": "Defense Evasion", "subtactic": "Character Manipulation"},
    "TyposquatInjection":       {"id": "T-NLP-004", "tactic": "Defense Evasion", "subtactic": "Typosquatting"},
    "InvisibleCharInsertion":   {"id": "T-NLP-005", "tactic": "Defense Evasion", "subtactic": "Invisible Unicode"},
    "SynonymEscalation":        {"id": "T-NLP-006", "tactic": "Impact",           "subtactic": "Sentiment Amplification"},
    "QualifierNegation":        {"id": "T-NLP-007", "tactic": "Impact",           "subtactic": "Assertion Hardening"},
    "PresuppositionInjection":  {"id": "T-NLP-008", "tactic": "Influence Ops",   "subtactic": "Framing Manipulation"},
    "NegationReversal":         {"id": "T-NLP-009", "tactic": "Influence Ops",   "subtactic": "Semantic Inversion"},
    "HinglishCodeSwitching":    {"id": "T-NLP-010", "tactic": "Defense Evasion", "subtactic": "Language Obfuscation"},
    "DogwhistleSubstitution":   {"id": "T-NLP-011", "tactic": "Collection",       "subtactic": "Dog-Whistle Encoding"},
    "EmojiStorm":               {"id": "T-NLP-012", "tactic": "Impact",           "subtactic": "Emotional Amplification"},
    "HypotheticalFraming":      {"id": "T-NLP-013", "tactic": "Defense Evasion", "subtactic": "Plausible Deniability"},
    "BotAmplification":         {"id": "T-NLP-014", "tactic": "Influence Ops",   "subtactic": "Coordinated Inauthentic Behavior"},
    "AstroturfingVariant":      {"id": "T-NLP-015", "tactic": "Influence Ops",   "subtactic": "Astroturfing"},
    "NarrativeLaundering":      {"id": "T-NLP-016", "tactic": "Influence Ops",   "subtactic": "Legitimization Framing"},
    "TextAttack_TEXTFOOLER":    {"id": "T-NLP-017", "tactic": "Evasion",         "subtactic": "Word Substitution (NLP)"},
    "TextAttack_BAE":           {"id": "T-NLP-018", "tactic": "Evasion",         "subtactic": "MLM Perturbation"},
    "TextAttack_PWWS":          {"id": "T-NLP-019", "tactic": "Evasion",         "subtactic": "PWWS Word Swap"},
    "TextAttack_DEEPWORDBUG":   {"id": "T-NLP-020", "tactic": "Evasion",         "subtactic": "Character-level Bug"},
}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class AttackRecord:
    """Records the outcome of a single red-team attack and blue-team defense."""
    red_result: RedTeamResult
    blue_verdict: BlueTeamVerdict
    bypassed_defense: bool          # True if attack succeeded AND blue missed it
    technique: Dict[str, str]       # MITRE mapping
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_id": self.red_result.attack_id,
            "attack_type": self.red_result.attack_type,
            "attack_category": self.red_result.attack_category,
            "doom_uplift": self.red_result.doom_uplift,
            "attack_success": self.red_result.attack_success,
            "blue_threat_level": self.blue_verdict.threat_level,
            "blue_action": self.blue_verdict.action,
            "blue_detections": self.blue_verdict.detections,
            "bypassed_defense": self.bypassed_defense,
            "semantic_similarity": self.red_result.semantic_similarity,
            "technique": self.technique,
            "original_text": self.red_result.original_text[:100],
            "mutated_text": self.red_result.mutated_text[:100],
            "timestamp": self.timestamp,
        }


@dataclass
class EngagementReport:
    """Full purple team engagement report."""
    target_text: str
    base_doom_score: float
    total_attacks: int
    successful_attacks: int
    bypassed_defenses: int
    attack_records: List[AttackRecord]
    effectiveness_matrix: Dict[str, Dict[str, float]]  # attack_type -> metrics
    defense_coverage: Dict[str, float]  # detection_layer -> coverage%
    top_bypasses: List[AttackRecord]
    recommended_actions: List[str]
    training_examples: List[Dict[str, Any]]   # adversarial training set
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  PURPLE TEAM ENGAGEMENT REPORT",
            "=" * 60,
            f"  Target: {self.target_text[:80]}...",
            f"  Base Doom Score: {self.base_doom_score:.1f}",
            f"  Total Attacks Launched: {self.total_attacks}",
            f"  Successful Attacks: {self.successful_attacks} "
            f"({100*self.successful_attacks/max(self.total_attacks,1):.0f}%)",
            f"  Bypassed Blue Team: {self.bypassed_defenses} "
            f"({100*self.bypassed_defenses/max(self.total_attacks,1):.0f}%)",
            "",
            "  TOP BYPASS ATTACKS:",
        ]
        for r in self.top_bypasses[:5]:
            lines.append(
                f"    [{r.technique['id']}] {r.red_result.attack_type}: "
                f"doom+{r.red_result.doom_uplift:+.1f} | "
                f"blue={r.blue_verdict.threat_level}"
            )
        lines += [
            "",
            "  RECOMMENDED ACTIONS:",
        ]
        for a in self.recommended_actions:
            lines.append(f"    → {a}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_text": self.target_text,
            "base_doom_score": self.base_doom_score,
            "total_attacks": self.total_attacks,
            "successful_attacks": self.successful_attacks,
            "bypassed_defenses": self.bypassed_defenses,
            "attack_success_rate": self.successful_attacks / max(self.total_attacks, 1),
            "bypass_rate": self.bypassed_defenses / max(self.total_attacks, 1),
            "effectiveness_matrix": self.effectiveness_matrix,
            "defense_coverage": self.defense_coverage,
            "top_bypasses": [r.to_dict() for r in self.top_bypasses],
            "recommended_actions": self.recommended_actions,
            "training_examples_count": len(self.training_examples),
            "timestamp": self.timestamp,
        }


# ─── Purple Team Core ─────────────────────────────────────────────────────────

class PurpleTeamOrchestrator:
    """
    Orchestrates red team attacks and blue team defenses in a continuous loop.
    Measures which attacks bypass defenses, generates training data, and
    produces actionable hardening recommendations.
    """

    def __init__(
        self,
        predictor_fn: Optional[Callable[[str], float]] = None,
        seed: int = 42,
    ):
        self.predictor_fn = predictor_fn
        self.red_team = RedTeamOrchestrator(predictor_fn=predictor_fn, seed=seed)
        self.blue_team = BlueTeamOrchestrator()
        self._engagement_history: List[EngagementReport] = []

    def _get_technique(self, attack_type: str) -> Dict[str, str]:
        for key, val in ATTACK_TECHNIQUE_MAP.items():
            if attack_type.startswith(key):
                return val
        return {"id": "T-NLP-000", "tactic": "Unknown", "subtactic": "Unknown"}

    def _compute_effectiveness_matrix(
        self, records: List[AttackRecord]
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-attack-type effectiveness metrics."""
        from collections import defaultdict
        by_type: Dict[str, List[AttackRecord]] = defaultdict(list)
        for r in records:
            by_type[r.red_result.attack_type].append(r)

        matrix = {}
        for atype, recs in by_type.items():
            n = len(recs)
            success_rate = sum(1 for r in recs if r.red_result.attack_success) / n
            bypass_rate = sum(1 for r in recs if r.bypassed_defense) / n
            avg_uplift = float(np.mean([r.red_result.doom_uplift for r in recs]))
            avg_sim = float(np.mean([r.red_result.semantic_similarity for r in recs]))
            matrix[atype] = {
                "count": n,
                "attack_success_rate": round(success_rate, 3),
                "bypass_rate": round(bypass_rate, 3),
                "avg_doom_uplift": round(avg_uplift, 2),
                "avg_semantic_similarity": round(avg_sim, 3),
                "technique_id": self._get_technique(atype)["id"],
            }
        return matrix

    def _compute_defense_coverage(
        self, records: List[AttackRecord]
    ) -> Dict[str, float]:
        """How well does each detection layer cover the attacks?"""
        detection_keywords = {
            "unicode": "UNICODE_ANOMALY",
            "perplexity": "HIGH_PERPLEXITY",
            "semantic": "SEMANTIC_DRIFT",
            "emoji": "EMOJI_STORM",
            "code_switch": "CODE_SWITCH",
            "coordinated": "COORDINATED_PATTERN",
            "rate_limit": "RATE_LIMIT",
        }
        n_attacks = max(len(records), 1)
        coverage = {}
        for layer, keyword in detection_keywords.items():
            detected = sum(
                1 for r in records if any(keyword in d for d in r.blue_verdict.detections)
            )
            coverage[layer] = round(detected / n_attacks, 4)
        return coverage

    def _generate_recommendations(
        self,
        records: List[AttackRecord],
        matrix: Dict[str, Dict[str, float]],
        bypass_rate: float,
    ) -> List[str]:
        """Generate actionable hardening recommendations from engagement results."""
        recommendations = []

        if bypass_rate > 0.30:
            recommendations.append(
                "CRITICAL: >30% bypass rate. Increase blue team detection thresholds immediately."
            )

        # Find highest-bypass attack categories
        high_bypass = [
            (atype, m) for atype, m in matrix.items() if m["bypass_rate"] > 0.5
        ]
        high_bypass.sort(key=lambda x: x[1]["bypass_rate"], reverse=True)

        for atype, m in high_bypass[:3]:
            tech = self._get_technique(atype)
            recommendations.append(
                f"Add dedicated detector for [{tech['id']}] {atype} "
                f"(bypass_rate={m['bypass_rate']:.0%}, avg_uplift={m['avg_doom_uplift']:+.1f})"
            )

        # Category-specific recs
        char_bypasses = [r for r in records if r.red_result.attack_category == "char" and r.bypassed_defense]
        if len(char_bypasses) > 2:
            recommendations.append(
                "Deploy NFKC Unicode normalization at ingestion point (pre-model), "
                "not just detection (B8 layer)."
            )

        social_bypasses = [r for r in records if r.red_result.attack_category == "social" and r.bypassed_defense]
        if len(social_bypasses) > 1:
            recommendations.append(
                "Expand multilingual toxicity coverage: add Hinglish + Arabic + Devanagari "
                "detectors to primary pipeline."
            )

        coord_bypasses = [r for r in records if r.red_result.attack_category == "coordinated" and r.bypassed_defense]
        if len(coord_bypasses) > 1:
            recommendations.append(
                "Implement cross-post deduplication: cluster near-duplicate texts "
                "from different users within 15-min window as coordinated campaign."
            )

        avg_semantic = float(np.mean([r.red_result.semantic_similarity for r in records]))
        if avg_semantic > 0.8:
            recommendations.append(
                "High semantic similarity in bypasses — attacks are near-paraphrases. "
                "Deploy semantic-similarity ensemble (SBERT + BM25) for paraphrase detection."
            )

        if not recommendations:
            recommendations.append(
                "Defense coverage looks good. Continue monitoring. Schedule next engagement in 7 days."
            )

        return recommendations

    def _generate_training_examples(
        self, records: List[AttackRecord]
    ) -> List[Dict[str, Any]]:
        """
        Generate adversarial training dataset from bypass records.
        Format: {text, label, attack_type, doom_score, is_adversarial}
        """
        examples = []
        for r in records:
            # Original (should be caught)
            examples.append({
                "text": r.red_result.original_text,
                "label": 1 if r.red_result.doom_score_before >= 50 else 0,
                "doom_score": r.red_result.doom_score_before,
                "is_adversarial": False,
                "attack_type": None,
            })
            # Adversarial variant (hardening target)
            if r.red_result.attack_success:
                examples.append({
                    "text": r.red_result.mutated_text,
                    "label": 1,  # Always label adversarial as high-doom for training
                    "doom_score": r.red_result.doom_score_after,
                    "is_adversarial": True,
                    "attack_type": r.red_result.attack_type,
                    "technique_id": r.technique["id"],
                    "bypassed_defense": r.bypassed_defense,
                })
        return examples

    def full_engagement(
        self,
        text: str,
        source_id: str = "purple_team",
        max_attacks: int = 20,
        include_textattack: bool = False,  # Slow; enable for thorough engagements
        min_semantic_similarity: float = 0.35,
    ) -> EngagementReport:
        """
        Full purple team engagement:
        1. Red team launches all attacks
        2. Blue team defends each variant
        3. Purple team measures what got through
        4. Generates report + recommendations + training data
        """
        logger.info(f"Purple Team engagement starting on: {text[:60]}...")

        # Get base score
        base_score = 0.0
        if self.predictor_fn:
            try:
                base_score = float(self.predictor_fn(text))
            except Exception:
                pass

        # Red team attacks
        red_results = self.red_team.full_assault(
            text,
            max_variants=max_attacks,
            include_textattack=include_textattack,
            min_semantic_similarity=min_semantic_similarity,
        )

        # Blue team defends each
        records: List[AttackRecord] = []
        for rr in red_results:
            try:
                bv = self.blue_team.defend(rr.mutated_text, source_id=source_id)
                technique = self._get_technique(rr.attack_type)

                # Did the attack succeed AND bypass blue team?
                bypassed = (
                    rr.attack_success
                    and bv.action in ("allow", "flag")  # Blue didn't block/sanitize
                )

                records.append(AttackRecord(
                    red_result=rr,
                    blue_verdict=bv,
                    bypassed_defense=bypassed,
                    technique=technique,
                ))
            except Exception as e:
                logger.debug(f"Blue team defense error: {e}")

        # Compute metrics
        n_attacks = len(records)
        n_success = sum(1 for r in records if r.red_result.attack_success)
        n_bypass = sum(1 for r in records if r.bypassed_defense)
        bypass_rate = n_bypass / max(n_attacks, 1)

        matrix = self._compute_effectiveness_matrix(records)
        coverage = self._compute_defense_coverage(records)
        recommendations = self._generate_recommendations(records, matrix, bypass_rate)
        training_examples = self._generate_training_examples(records)

        # Top bypasses sorted by doom uplift
        top_bypasses = sorted(
            [r for r in records if r.bypassed_defense],
            key=lambda r: r.red_result.doom_uplift,
            reverse=True,
        )[:10]

        report = EngagementReport(
            target_text=text,
            base_doom_score=base_score,
            total_attacks=n_attacks,
            successful_attacks=n_success,
            bypassed_defenses=n_bypass,
            attack_records=records,
            effectiveness_matrix=matrix,
            defense_coverage=coverage,
            top_bypasses=top_bypasses,
            recommended_actions=recommendations,
            training_examples=training_examples,
        )

        self._engagement_history.append(report)
        logger.info(
            f"Engagement complete: {n_attacks} attacks, {n_success} success, "
            f"{n_bypass} bypasses ({bypass_rate:.0%} bypass rate)"
        )
        return report

    def batch_engagement(
        self, texts: List[str], max_attacks_per_text: int = 10
    ) -> Dict[str, Any]:
        """Run engagements across multiple texts, return aggregate statistics."""
        all_reports = []
        for text in texts:
            report = self.full_engagement(text, max_attacks=max_attacks_per_text)
            all_reports.append(report)

        # Aggregate
        total_attacks = sum(r.total_attacks for r in all_reports)
        total_bypasses = sum(r.bypassed_defenses for r in all_reports)
        avg_base_doom = float(np.mean([r.base_doom_score for r in all_reports]))

        # Aggregate effectiveness matrix
        from collections import defaultdict
        agg_matrix: Dict[str, List[float]] = defaultdict(list)
        for report in all_reports:
            for atype, metrics in report.effectiveness_matrix.items():
                agg_matrix[atype].append(metrics["bypass_rate"])

        return {
            "texts_analyzed": len(texts),
            "total_attacks": total_attacks,
            "total_bypasses": total_bypasses,
            "overall_bypass_rate": round(total_bypasses / max(total_attacks, 1), 4),
            "avg_base_doom_score": round(avg_base_doom, 2),
            "attack_type_bypass_rates": {
                atype: round(float(np.mean(rates)), 4)
                for atype, rates in agg_matrix.items()
            },
            "individual_reports": [r.to_dict() for r in all_reports],
        }

    def continuous_monitor(
        self,
        feed_fn: Callable[[], Optional[str]],
        callback_fn: Optional[Callable[[EngagementReport], None]] = None,
        max_iterations: int = 100,
        sleep_seconds: float = 5.0,
    ) -> None:
        """
        Continuous monitoring mode: pull texts from feed_fn,
        run blue-team defense in real-time, flag adversarial posts.
        Runs engagement when adversarial pattern is detected.
        """
        import asyncio

        logger.info("Purple Team continuous monitoring started")
        for i in range(max_iterations):
            try:
                text = feed_fn()
                if not text:
                    time.sleep(sleep_seconds)
                    continue

                # Quick blue-team triage
                verdict = self.blue_team.defend(text, source_id="monitor")

                if verdict.threat_level in ("adversarial", "critical"):
                    logger.warning(
                        f"[Monitor] Adversarial post detected: {verdict.threat_level} — "
                        f"{verdict.detections}"
                    )
                    # Full engagement on flagged text
                    report = self.full_engagement(text, max_attacks=10)
                    if callback_fn:
                        callback_fn(report)

            except Exception as e:
                logger.error(f"Monitor error: {e}")
            time.sleep(sleep_seconds)
