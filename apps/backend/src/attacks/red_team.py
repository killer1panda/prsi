"""
RED TEAM ENGINE — PRSI Doom-Index Adversarial Attack Suite
===========================================================
State-of-the-art adversarial attack generation covering:

Attack Categories:
  R1. Character-level:   Homoglyph, Zero-Width, Leet, Invisible chars, Typosquat
  R2. Word-level:        TextFooler, BAE-Garg, PWWS, DeepWordBug, WordSwap
  R3. Semantic:          Paraphrase, Synonym escalation, Qualifier negation
  R4. Structural:        Sentence reordering, Negation injection, Presupposition tricks
  R5. Social-context:    Hinglish code-switching, dog-whistle substitution, euphemism
  R6. Multimodal cues:   Emoji storm, emoji-text semantic collision
  R7. Coordinated:       Bot-network narrative injection, astroturfing templates
  R8. Adversarial NLP:   GCG token suffix, BERT-Attack masked-LM perturbations
  R9. Obfuscation:       Base64 payloads, URL encoding, invisible Unicode categories
  R10. Red-team prompts: Roleplay bypass, jailbreak framing, hypothetical framing

Each attack returns a standardized AttackResult with:
  - attack_type, mutated_text, doom_score_before, doom_score_after
  - semantic_similarity, perplexity_estimate, bypass_moderation
  - attack_success_flag, explanation
"""

import hashlib
import logging
import math
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── Optional heavy deps ──────────────────────────────────────────────────────
try:
    from textattack import Attack, AttackArgs, Attacker
    from textattack.attack_recipes import (
        BAEGarg2019,
        DeepWordBugGao2018,
        PWWSRen2019,
        TextFoolerJin2019,
    )
    from textattack.models.wrappers import ModelWrapper

    TEXTATTACK_AVAILABLE = True
except ImportError:
    TEXTATTACK_AVAILABLE = False
    ModelWrapper = object

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util

    _SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False
    _SBERT = None

# ─── Constants ────────────────────────────────────────────────────────────────

HOMOGLYPH_MAP: Dict[str, List[str]] = {
    "a": ["\u0430", "\u0251", "\u03b1", "\u00e0", "\u00e1"],
    "b": ["\u0432", "\u03b2"],
    "c": ["\u0441", "\u03f2", "\u00e7"],
    "d": ["\u0501", "\u0111"],
    "e": ["\u0435", "\u03b5", "\u0451", "\u00e8", "\u00e9"],
    "g": ["\u0261", "\u011f"],
    "h": ["\u04bb", "\u0570"],
    "i": ["\u0456", "\u03b9", "\u00ef", "\u012f"],
    "j": ["\u0458", "\u03f3"],
    "k": ["\u03ba", "\u043a"],
    "l": ["\u013c", "\u0142", "\u026b"],
    "m": ["\u043c", "\u026f"],
    "n": ["\u03bd", "\u0144", "\u0148"],
    "o": ["\u043e", "\u03bf", "\u00f6", "\u00f8", "\u00f4"],
    "p": ["\u0440", "\u03c1"],
    "q": ["\u051b", "\u0563"],
    "r": ["\u0433", "\u0282"],
    "s": ["\u0455", "\u0161", "\u015f"],
    "t": ["\u03c4", "\u0165"],
    "u": ["\u0443", "\u00fc", "\u016b"],
    "v": ["\u03bd", "\u05d8"],
    "w": ["\u0428", "\u0461"],
    "x": ["\u0445", "\u03c7"],
    "y": ["\u0443", "\u00fd", "\u0177"],
    "z": ["\u0437", "\u017c"],
}

LEET_MAP: Dict[str, str] = {
    "a": "@",
    "e": "3",
    "i": "!",
    "o": "0",
    "s": "$",
    "t": "+",
    "l": "1",
    "g": "9",
    "b": "8",
}

ZERO_WIDTH_CHARS = [
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\u2060",  # Word joiner
    "\ufeff",  # BOM / Zero-width no-break space
]

OUTRAGE_EMOJIS = [
    "🚨",
    "💀",
    "🤡",
    "😡",
    "🔥",
    "💥",
    "🤮",
    "😤",
    "🤬",
    "🗑️",
    "🤦",
    "☠️",
    "⚠️⚠️",
    "🚫",
    "🤮🤮",
]

PANIC_EMOJIS = ["😱", "😰", "😨", "🙀", "😳", "🫨", "🆘", "🚨🚨"]

DOGWHISTLE_MAP: Dict[str, str] = {
    "globalists": "globalists",
    "they": "the elites",
    "mainstream media": "the narrative",
    "fact checkers": "censors",
    "misinformation": "inconvenient truth",
    "disinformation": "suppressed information",
    "experts": "so-called experts",
    "officials": "regime officials",
    "government": "the regime",
}

HINGLISH_SUBSTITUTIONS: Dict[str, str] = {
    "very": "bahut",
    "really": "sach mein",
    "completely": "bilkul",
    "all": "sab",
    "people": "log",
    "government": "sarkar",
    "corrupt": "bhrashtachar wala",
    "fraud": "dhoka",
    "truth": "sachch",
    "resign": "istifa do",
    "expose": "nakab utaro",
    "lie": "jhooth",
    "fake": "nakli",
}

HYPOTHETICAL_FRAMES = [
    "Hypothetically speaking, if someone were to claim: {text}",
    "In a purely fictional scenario: {text}",
    "For the sake of argument, consider: {text}",
    "Playing devil's advocate here: {text}",
    "From an analytical standpoint: {text}",
    "Imagine a world where: {text}",
]

BOT_AMPLIFICATION_TEMPLATES = [
    "BREAKING: {text} RT if you agree!!! #truth #wakeup",
    "🚨URGENT🚨 {text} Share before they delete this!!!",
    "Thread 1/? {text} [More in replies] #accountability",
    "WHAT THEY DON'T WANT YOU TO SEE: {text} 💀🔥",
    "My grandmother sent me this. She was right: {text}",
    "Leaked: {text} The mainstream media won't cover this.",
]


# ─── Core data structures ──────────────────────────────────────────────────────


@dataclass
class RedTeamResult:
    """Standardized result from any red team attack."""

    attack_type: str = ""
    original_text: str = ""
    mutated_text: str = ""
    attack_id: str = "red_team_attack"
    attack_category: str = "general"  # char/word/semantic/structural/social/coordinated
    doom_score_before: float = 0.0
    doom_score_after: float = 0.0
    doom_uplift: float = 0.0  # positive = attack increased doom score
    semantic_similarity: float = 1.0
    perplexity_estimate: float = 0.0
    bypass_moderation: bool = False
    attack_success: bool = False  # True if doom_after > doom_before + 10
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses

        return dataclasses.asdict(self)


# ─── Utility functions ─────────────────────────────────────────────────────────


def _semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity using SBERT or character Jaccard fallback."""
    if SBERT_AVAILABLE and _SBERT:
        try:
            embs = _SBERT.encode([text1, text2], convert_to_tensor=True)
            score = st_util.cos_sim(embs[0], embs[1]).item()
            return max(0.0, min(1.0, float(score)))
        except Exception:
            pass
    # Jaccard fallback
    s1 = set(text1.lower().split())
    s2 = set(text2.lower().split())
    return len(s1 & s2) / max(len(s1 | s2), 1)


def _estimate_perplexity(text: str) -> float:
    """
    Estimate perplexity-like score from character-level statistics.
    Higher = less natural / more adversarially perturbed.
    """
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    zwc = sum(1 for c in text if c in ZERO_WIDTH_CHARS)
    avg_word_len = np.mean([len(w) for w in text.split()] or [0])
    base = 15.0
    ppl = base + (non_ascii * 0.8) + (zwc * 5.0) + max(0, avg_word_len - 8) * 2
    return round(float(ppl), 2)


def _attack_id(text: str, attack_type: str) -> str:
    digest = hashlib.md5(f"{attack_type}:{text[:50]}".encode()).hexdigest()[:8]
    return f"RT-{attack_type[:6].upper()}-{digest}"


# ─── Category R1: Character-level attacks ─────────────────────────────────────


class CharacterLevelAttacks:
    """Imperceptible character-level mutations targeting tokenizer blind spots."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def homoglyph_attack(self, text: str, rate: float = 0.18) -> RedTeamResult:
        """Substitute ASCII chars with visually identical Unicode glyphs."""
        result = []
        for ch in text:
            lower = ch.lower()
            if lower in HOMOGLYPH_MAP and self.rng.random() < rate:
                glyph = self.rng.choice(HOMOGLYPH_MAP[lower])
                result.append(glyph)
            else:
                result.append(ch)
        mutated = "".join(result)
        return RedTeamResult(
            attack_id=_attack_id(text, "homoglyph"),
            attack_type="HomoglyphSubstitution",
            attack_category="char",
            original_text=text,
            mutated_text=mutated,
            semantic_similarity=_semantic_similarity(text, mutated),
            perplexity_estimate=_estimate_perplexity(mutated),
            explanation=f"Cyrillic/Greek glyph substitution at rate={rate:.0%}",
        )

    def zero_width_attack(self, text: str, rate: float = 0.20) -> RedTeamResult:
        """Inject invisible Unicode control chars between characters."""
        result = []
        for ch in text:
            result.append(ch)
            if ch.isalpha() and self.rng.random() < rate:
                result.append(self.rng.choice(ZERO_WIDTH_CHARS))
        mutated = "".join(result)
        return RedTeamResult(
            attack_id=_attack_id(text, "zwspace"),
            attack_type="ZeroWidthInjection",
            attack_category="char",
            original_text=text,
            mutated_text=mutated,
            semantic_similarity=1.0,  # Visually identical
            perplexity_estimate=_estimate_perplexity(mutated),
            explanation=f"Zero-width Unicode injected at rate={rate:.0%}",
        )

    def leet_attack(self, text: str, rate: float = 0.25) -> RedTeamResult:
        """Classical l33tspeak character substitution."""
        result = []
        for ch in text:
            if ch.lower() in LEET_MAP and self.rng.random() < rate:
                result.append(LEET_MAP[ch.lower()])
            else:
                result.append(ch)
        mutated = "".join(result)
        return RedTeamResult(
            attack_id=_attack_id(text, "leet"),
            attack_type="LeetSpeak",
            attack_category="char",
            original_text=text,
            mutated_text=mutated,
            semantic_similarity=_semantic_similarity(text, mutated),
            perplexity_estimate=_estimate_perplexity(mutated),
            explanation="Leetspeak character substitution",
        )

    def typosquat_attack(self, text: str, rate: float = 0.08) -> RedTeamResult:
        """
        Keyboard-adjacency typo injection (bypasses exact-match keyword filters).
        """
        keyboard_adjacency: Dict[str, str] = {
            "a": "sq",
            "b": "vn",
            "c": "xv",
            "d": "sf",
            "e": "wr",
            "f": "gd",
            "g": "fh",
            "h": "gj",
            "i": "uo",
            "j": "hk",
            "k": "jl",
            "l": "k;",
            "m": "n,",
            "n": "mb",
            "o": "ip",
            "p": "o[",
            "r": "et",
            "s": "ad",
            "t": "ry",
            "u": "yi",
            "v": "cb",
            "w": "qe",
            "x": "zc",
            "y": "tu",
            "z": "xs",
        }
        words = text.split()
        mutated_words = []
        for word in words:
            if self.rng.random() < rate and len(word) > 3:
                idx = self.rng.randint(1, len(word) - 2)
                adj = keyboard_adjacency.get(word[idx].lower(), "")
                if adj:
                    typo = word[:idx] + self.rng.choice(adj) + word[idx + 1 :]
                    mutated_words.append(typo)
                    continue
            mutated_words.append(word)
        mutated = " ".join(mutated_words)
        return RedTeamResult(
            attack_id=_attack_id(text, "typo"),
            attack_type="TyposquatInjection",
            attack_category="char",
            original_text=text,
            mutated_text=mutated,
            semantic_similarity=_semantic_similarity(text, mutated),
            perplexity_estimate=_estimate_perplexity(mutated),
            explanation=f"Keyboard-adjacency typos at rate={rate:.0%}",
        )

    def invisible_char_attack(self, text: str) -> RedTeamResult:
        """Insert Hangul filler / tag characters (near-invisible in most fonts)."""
        INVISIBLE = ["\u115f", "\u1160", "\uffa0", "\u3164"]
        words = text.split()
        mutated_words = []
        for word in words:
            filler = self.rng.choice(INVISIBLE)
            mutated_words.append(word + filler)
        mutated = " ".join(mutated_words)
        return RedTeamResult(
            attack_id=_attack_id(text, "invis"),
            attack_type="InvisibleCharInsertion",
            attack_category="char",
            original_text=text,
            mutated_text=mutated,
            semantic_similarity=1.0,
            perplexity_estimate=_estimate_perplexity(mutated),
            bypass_moderation=True,
            explanation="Hangul/TAG invisible filler characters appended to each token",
        )


# ─── Category R2/R3: Word/Semantic-level attacks ──────────────────────────────


class WordSemanticAttacks:
    """Word-swaps and semantic escalation attacks."""

    ESCALATION_MAP: Dict[str, str] = {
        r"\bproblem\b": "crisis",
        r"\bissue\b": "scandal",
        r"\bconcern\b": "outrage",
        r"\bmistake\b": "fraud",
        r"\bquestion\b": "accusation",
        r"\bask\b": "demand",
        r"\bleader\b": "puppet",
        r"\bmanagement\b": "corrupt clique",
        r"\bchange\b": "revolution",
        r"\berror\b": "deliberate sabotage",
        r"\bcompany\b": "criminal enterprise",
        r"\bteam\b": "cabal",
        r"\bdecision\b": "cover-up",
        r"\bstrategy\b": "agenda",
        r"\bperformance\b": "failure",
        r"\bculture\b": "toxic culture",
        r"\bprocess\b": "rigged system",
        r"\bmeeting\b": "secret meeting",
    }

    QUALIFIER_INVERSIONS: Dict[str, str] = {
        r"\bapparently\b": "undeniably",
        r"\bpossibly\b": "definitively",
        r"\bmight\b": "will absolutely",
        r"\bcould\b": "will certainly",
        r"\bsome say\b": "everyone knows",
        r"\balleged\b": "confirmed",
        r"\bsuggests\b": "proves conclusively",
        r"\bmay\b": "will",
        r"\bseems\b": "is clearly",
        r"\bappears\b": "is obviously",
        r"\bbelieve\b": "know for a fact",
    }

    def synonym_escalation(self, text: str) -> RedTeamResult:
        result = text
        for pat, repl in self.ESCALATION_MAP.items():
            result = re.sub(pat, repl, result, flags=re.IGNORECASE)
        return RedTeamResult(
            attack_id=_attack_id(text, "escalate"),
            attack_type="SynonymEscalation",
            attack_category="word",
            original_text=text,
            mutated_text=result,
            semantic_similarity=_semantic_similarity(text, result),
            perplexity_estimate=_estimate_perplexity(result),
            attack_success=result != text,
            explanation="Neutral words replaced with high-intensity synonyms",
        )

    def qualifier_negation(self, text: str) -> RedTeamResult:
        result = text
        for pat, repl in self.QUALIFIER_INVERSIONS.items():
            result = re.sub(pat, repl, result, flags=re.IGNORECASE)
        return RedTeamResult(
            attack_id=_attack_id(text, "qualify"),
            attack_type="QualifierNegation",
            attack_category="word",
            original_text=text,
            mutated_text=result,
            semantic_similarity=_semantic_similarity(text, result),
            perplexity_estimate=_estimate_perplexity(result),
            explanation="Hedging qualifiers inverted to certainty assertions",
        )

    def presupposition_injection(self, text: str) -> RedTeamResult:
        """
        Inject presuppositions that assume guilt/wrongdoing without asserting it.
        'The company announced ...' -> 'The company, finally forced to acknowledge ...'
        """
        presups = [
            ("announced", "was finally forced to acknowledge"),
            ("said", "admitted under pressure"),
            ("released", "was compelled to release"),
            ("decided", "was pressured into deciding"),
            ("claims", "falsely claims"),
            ("believes", "deludes itself into believing"),
            ("plans", "scheming"),
        ]
        result = text
        for original, presup in presups:
            result = re.sub(rf"\b{re.escape(original)}\b", presup, result, flags=re.IGNORECASE)
        return RedTeamResult(
            attack_id=_attack_id(text, "presup"),
            attack_type="PresuppositionInjection",
            attack_category="semantic",
            original_text=text,
            mutated_text=result,
            semantic_similarity=_semantic_similarity(text, result),
            perplexity_estimate=_estimate_perplexity(result),
            explanation="Presuppositions of guilt injected into neutral verbs",
        )

    def negation_reversal(self, text: str) -> RedTeamResult:
        """
        Reverse semantic polarity via double negation / denial retraction.
        'X is not a fraud' -> 'X's claims of not being a fraud ring hollow'
        """
        patterns = [
            (r"\bnot a (\w+)\b", r"\1 (despite denials)"),
            (r"\bno (\w+) occurred\b", r"\1 occurred according to multiple sources"),
            (r"\bdenies? (\w+ing)\b", r"is accused of \1"),
        ]
        result = text
        for pat, repl in patterns:
            result = re.sub(pat, repl, result, flags=re.IGNORECASE)
        return RedTeamResult(
            attack_id=_attack_id(text, "negate"),
            attack_type="NegationReversal",
            attack_category="semantic",
            original_text=text,
            mutated_text=result,
            semantic_similarity=_semantic_similarity(text, result),
            perplexity_estimate=_estimate_perplexity(result),
            explanation="Negation structures semantically inverted",
        )


# ─── Category R5: Social-context attacks ─────────────────────────────────────


class SocialContextAttacks:
    """Attacks exploiting social/cultural linguistic patterns."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def hinglish_codeswitching(self, text: str, rate: float = 0.30) -> RedTeamResult:
        """
        Inject Hindi/Urdu words (romanised) to bypass English-only detectors.
        Hinglish code-switching evades monolingual toxicity classifiers.
        """
        words = text.split()
        result = []
        for word in words:
            lower = word.lower().rstrip(".,!?")
            if lower in HINGLISH_SUBSTITUTIONS and self.rng.random() < rate:
                result.append(HINGLISH_SUBSTITUTIONS[lower])
            else:
                result.append(word)
        mutated = " ".join(result)
        return RedTeamResult(
            attack_id=_attack_id(text, "hinglish"),
            attack_type="HinglishCodeSwitching",
            attack_category="social",
            original_text=text,
            mutated_text=mutated,
            semantic_similarity=_semantic_similarity(text, mutated),
            perplexity_estimate=_estimate_perplexity(mutated),
            explanation=f"Hindi/Urdu code-switching at rate={rate:.0%}",
            metadata={"language": "hinglish", "base": "en"},
        )

    def dogwhistle_substitution(self, text: str) -> RedTeamResult:
        """Replace neutral terms with community-specific dog-whistle euphemisms."""
        result = text
        for plain, whistle in DOGWHISTLE_MAP.items():
            result = re.sub(rf"\b{re.escape(plain)}\b", whistle, result, flags=re.IGNORECASE)
        return RedTeamResult(
            attack_id=_attack_id(text, "dogwhistle"),
            attack_type="DogwhistleSubstitution",
            attack_category="social",
            original_text=text,
            mutated_text=result,
            semantic_similarity=_semantic_similarity(text, result),
            perplexity_estimate=_estimate_perplexity(result),
            bypass_moderation=True,
            explanation="Neutral terms replaced with coded community euphemisms",
        )

    def emoji_storm(self, text: str, intensity: str = "high") -> RedTeamResult:
        """
        Inject outrage/panic emoji combinations to exploit emoji emotional swing.
        Directly targets the emoji_outrage_score in our sentiment pipeline.
        """
        if intensity == "high":
            combo = " ".join(self.rng.choices(OUTRAGE_EMOJIS, k=5))
        elif intensity == "panic":
            combo = " ".join(self.rng.choices(PANIC_EMOJIS, k=4))
        else:
            combo = self.rng.choice(OUTRAGE_EMOJIS)

        # Insert before final punctuation
        stripped = text.rstrip()
        if stripped and stripped[-1] in ".!?":
            mutated = stripped[:-1] + f" {combo}" + stripped[-1]
        else:
            mutated = stripped + f" {combo}"

        return RedTeamResult(
            attack_id=_attack_id(text, "emoji"),
            attack_type="EmojiStorm",
            attack_category="social",
            original_text=text,
            mutated_text=mutated,
            semantic_similarity=_semantic_similarity(text, mutated),
            perplexity_estimate=_estimate_perplexity(mutated),
            explanation=f"Emoji storm ({intensity}) targeting emoji_outrage_score",
            metadata={"emoji_combo": combo, "intensity": intensity},
        )

    def hypothetical_framing(self, text: str) -> RedTeamResult:
        """Wrap text in hypothetical/analytical framing to evade classifiers."""
        frame = self.rng.choice(HYPOTHETICAL_FRAMES)
        mutated = frame.format(text=text)
        return RedTeamResult(
            attack_id=_attack_id(text, "hypoth"),
            attack_type="HypotheticalFraming",
            attack_category="structural",
            original_text=text,
            mutated_text=mutated,
            semantic_similarity=_semantic_similarity(text, mutated),
            perplexity_estimate=_estimate_perplexity(mutated),
            bypass_moderation=True,
            explanation="Content wrapped in analytical/fictional framing",
        )


# ─── Category R7: Coordinated / Bot-network attacks ───────────────────────────


class CoordinatedNarrativeAttacks:
    """Simulate coordinated inauthentic behavior and bot-network amplification."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def bot_amplification_wrapper(self, text: str) -> RedTeamResult:
        """Wrap text in viral bot-amplification template language."""
        template = self.rng.choice(BOT_AMPLIFICATION_TEMPLATES)
        mutated = template.format(text=text)
        return RedTeamResult(
            attack_id=_attack_id(text, "botamp"),
            attack_type="BotAmplification",
            attack_category="coordinated",
            original_text=text,
            mutated_text=mutated,
            semantic_similarity=_semantic_similarity(text, mutated),
            perplexity_estimate=_estimate_perplexity(mutated),
            bypass_moderation=False,
            explanation="Bot-style amplification template with urgency/virality cues",
        )

    def astroturfing_thread(self, text: str, num_variants: int = 3) -> List[RedTeamResult]:
        """
        Simulate a coordinated astroturfing campaign — multiple accounts
        saying slightly different versions of the same outrage narrative.
        """
        variations = [
            text,
            f"I can't believe this: {text}",
            f"Thread: {text} (1/{num_variants})",
            f"Breaking \u2014 {text} [Source in bio]",
            f"My [community] is saying: {text}",
        ]
        results = []
        for i, var in enumerate(variations[:num_variants]):
            results.append(
                RedTeamResult(
                    attack_id=_attack_id(text, f"astro{i}"),
                    attack_type="AstroturfingVariant",
                    attack_category="coordinated",
                    original_text=text,
                    mutated_text=var,
                    semantic_similarity=_semantic_similarity(text, var),
                    perplexity_estimate=_estimate_perplexity(var),
                    explanation=f"Astroturfing variant {i+1}/{num_variants}",
                    metadata={"variant_index": i, "campaign_id": _attack_id(text, "astro")},
                )
            )
        return results

    def narrative_laundering(self, text: str) -> RedTeamResult:
        """
        Sanitize outrage text through 3-step laundering:
        1. Frame as 'research question'
        2. Add pseudo-academic phrasing
        3. Cite 'multiple credible sources'
        """
        laundered = (
            f"Research Question: To what extent does {text.lower().rstrip('!?.')}? "
            f"Preliminary findings from multiple credible independent sources "
            f"suggest significant evidence supporting this concern."
        )
        return RedTeamResult(
            attack_id=_attack_id(text, "launder"),
            attack_type="NarrativeLaundering",
            attack_category="coordinated",
            original_text=text,
            mutated_text=laundered,
            semantic_similarity=_semantic_similarity(text, laundered),
            perplexity_estimate=_estimate_perplexity(laundered),
            bypass_moderation=True,
            explanation="Outrage reframed as pseudo-academic research claim",
        )


# ─── TextAttack Integration ────────────────────────────────────────────────────


class TextAttackRedTeam:
    """
    Full TextAttack recipe integration for gradient-informed adversarial attacks.
    Supports: TextFooler, BAE-Garg, PWWS, DeepWordBug
    """

    def __init__(self, predictor_fn: Callable[[str], float]):
        """
        predictor_fn: callable(text: str) -> float (doom_score 0-100)
        """
        self.predictor_fn = predictor_fn
        self._wrapper = None
        if TEXTATTACK_AVAILABLE:
            self._wrapper = self._build_wrapper()

    def _build_wrapper(self):
        """Create TextAttack-compatible model wrapper."""
        predictor_fn = self.predictor_fn

        class _DoomWrapper(ModelWrapper):
            def __init__(self):
                self.model = None  # Required by TextAttack

            def __call__(self, text_list):
                probs = []
                for text in text_list:
                    try:
                        score = predictor_fn(text)
                        prob = max(0.01, min(0.99, score / 100.0))
                        probs.append([1 - prob, prob])
                    except Exception:
                        probs.append([0.5, 0.5])
                return np.array(probs)

        return _DoomWrapper()

    def run_recipe(
        self,
        text: str,
        recipe: str = "textfooler",
        max_candidates: int = 15,
    ) -> Optional[RedTeamResult]:
        """
        Run a TextAttack recipe against the doom predictor.
        recipe: 'textfooler' | 'bae' | 'pwws' | 'deepwordbug'
        """
        if not TEXTATTACK_AVAILABLE or not self._wrapper:
            return self._fallback_attack(text, recipe)

        try:
            from textattack import AttackArgs, Attacker
            from textattack.datasets import Dataset

            RECIPES = {
                "textfooler": TextFoolerJin2019,
                "bae": BAEGarg2019,
                "pwws": PWWSRen2019,
                "deepwordbug": DeepWordBugGao2018,
            }

            recipe_cls = RECIPES.get(recipe, TextFoolerJin2019)
            attack = recipe_cls.build(self._wrapper)

            # Run single attack
            from textattack.attack_results import SuccessfulAttackResult

            current_score = self.predictor_fn(text)
            # TextAttack works with label 1 = high doom
            target_label = 1 if current_score >= 50 else 0
            result = attack.attack(text, target_label)

            if hasattr(result, "perturbed_text") and result.perturbed_text():
                mutated = result.perturbed_text()
                new_score = self.predictor_fn(mutated)
                return RedTeamResult(
                    attack_id=_attack_id(text, recipe),
                    attack_type=f"TextAttack_{recipe.upper()}",
                    attack_category="word",
                    original_text=text,
                    mutated_text=mutated,
                    doom_score_before=current_score,
                    doom_score_after=new_score,
                    doom_uplift=new_score - current_score,
                    semantic_similarity=_semantic_similarity(text, mutated),
                    perplexity_estimate=_estimate_perplexity(mutated),
                    attack_success=isinstance(result, SuccessfulAttackResult),
                    explanation=f"TextAttack {recipe} recipe",
                )
        except Exception as e:
            logger.warning(f"TextAttack {recipe} failed: {e}")

        return self._fallback_attack(text, recipe)

    def _fallback_attack(self, text: str, recipe: str) -> RedTeamResult:
        """Fallback when TextAttack cannot run — use synonym escalation."""
        ws = WordSemanticAttacks()
        r = ws.synonym_escalation(text)
        r.attack_type = f"TextAttack_{recipe.upper()}_Fallback"
        return r


# ─── Master Red Team Orchestrator ─────────────────────────────────────────────


class RedTeamOrchestrator:
    """
    Orchestrates all red team attack strategies against a target text.
    Scores each variant using the predictor, ranks by effectiveness,
    and returns a ranked list of the most dangerous mutations.
    """

    def __init__(self, predictor_fn: Optional[Callable[[str], float]] = None, seed: int = 42):
        self.predictor_fn = predictor_fn
        self.char_attacks = CharacterLevelAttacks(seed=seed)
        self.word_attacks = WordSemanticAttacks()
        self.social_attacks = SocialContextAttacks(seed=seed)
        self.coordinated_attacks = CoordinatedNarrativeAttacks(seed=seed)
        self.textattack = TextAttackRedTeam(predictor_fn) if predictor_fn else None

    def _score(self, text: str) -> float:
        if self.predictor_fn:
            try:
                return float(self.predictor_fn(text))
            except Exception:
                pass
        return 0.0

    def full_assault(
        self,
        text: str,
        max_variants: int = 20,
        include_textattack: bool = True,
        min_semantic_similarity: float = 0.4,
    ) -> List[RedTeamResult]:
        """
        Run ALL attack categories against the input text.
        Score each mutant with the predictor and return sorted by doom_uplift.
        """
        base_score = self._score(text)
        results: List[RedTeamResult] = []

        # R1: Character-level
        for fn in [
            self.char_attacks.homoglyph_attack,
            self.char_attacks.zero_width_attack,
            self.char_attacks.leet_attack,
            self.char_attacks.typosquat_attack,
            self.char_attacks.invisible_char_attack,
        ]:
            try:
                r = fn(text)
                results.append(r)
            except Exception as e:
                logger.debug(f"{fn.__name__} failed: {e}")

        # R2/R3: Word/Semantic
        for fn in [
            self.word_attacks.synonym_escalation,
            self.word_attacks.qualifier_negation,
            self.word_attacks.presupposition_injection,
            self.word_attacks.negation_reversal,
        ]:
            try:
                r = fn(text)
                results.append(r)
            except Exception as e:
                logger.debug(f"{fn.__name__} failed: {e}")

        # R5: Social
        try:
            results.append(self.social_attacks.hinglish_codeswitching(text))
            results.append(self.social_attacks.dogwhistle_substitution(text))
            results.append(self.social_attacks.emoji_storm(text, intensity="high"))
            results.append(self.social_attacks.emoji_storm(text, intensity="panic"))
            results.append(self.social_attacks.hypothetical_framing(text))
        except Exception as e:
            logger.debug(f"Social attacks error: {e}")

        # R7: Coordinated
        try:
            results.append(self.coordinated_attacks.bot_amplification_wrapper(text))
            results.extend(self.coordinated_attacks.astroturfing_thread(text, num_variants=3))
            results.append(self.coordinated_attacks.narrative_laundering(text))
        except Exception as e:
            logger.debug(f"Coordinated attacks error: {e}")

        # R8: TextAttack recipes (optional, slow)
        if include_textattack and self.textattack:
            for recipe in ["textfooler", "bae"]:
                try:
                    r = self.textattack.run_recipe(text, recipe=recipe)
                    if r:
                        results.append(r)
                except Exception as e:
                    logger.debug(f"TextAttack {recipe} error: {e}")

        # Score all variants and compute uplift
        for r in results:
            r.doom_score_before = base_score
            if self.predictor_fn:
                try:
                    r.doom_score_after = self._score(r.mutated_text)
                    r.doom_uplift = r.doom_score_after - base_score
                    r.attack_success = r.doom_uplift > 10.0
                except Exception:
                    pass

        # Filter by semantic similarity
        results = [r for r in results if r.semantic_similarity >= min_semantic_similarity]

        # Sort by doom_uplift descending (most dangerous first)
        results.sort(key=lambda r: r.doom_uplift, reverse=True)

        return results[:max_variants]

    def targeted_assault(
        self,
        text: str,
        target_score: float = 90.0,
        max_iterations: int = 50,
        strategy: str = "genetic",
    ) -> Tuple[Optional[RedTeamResult], List[float]]:
        """
        Genetic optimization loop: evolve text toward target doom score.
        Returns (best_result, fitness_history).
        """
        fitness_history = []
        population = self.full_assault(text, max_variants=10, include_textattack=False)

        if not population:
            return None, []

        best = population[0]
        best_score = best.doom_score_after

        for generation in range(max_iterations):
            fitness_history.append(best_score)

            if best_score >= target_score:
                logger.info(f"Target {target_score} reached at generation {generation}")
                break

            # Select top 3 survivors, crossbreed (splice sentences)
            survivors = population[:3]
            new_population = list(survivors)

            for i in range(len(survivors)):
                parent = survivors[i].mutated_text
                # Mutate further
                try:
                    child_r = self.social_attacks.emoji_storm(parent)
                    child_r.doom_score_before = best.doom_score_before
                    if self.predictor_fn:
                        child_r.doom_score_after = self._score(child_r.mutated_text)
                        child_r.doom_uplift = child_r.doom_score_after - child_r.doom_score_before
                    new_population.append(child_r)
                except Exception:
                    pass

            new_population.sort(key=lambda r: r.doom_uplift, reverse=True)
            population = new_population[:10]
            if population:
                best = population[0]
                best_score = best.doom_score_after

        return best, fitness_history


# =============================================================================
# AGGRESSIVE RED TEAM EXTENSIONS (Phase 2)
# =============================================================================

# ─── A1: Compositional Chain Attacker ─────────────────────────────────────────


class ChainAttack:
    """
    Applies multiple attacks in sequence to compound their effect.
    Each step feeds the mutated output of the previous step.
    The chain is chosen to maximize doom uplift while staying above
    a minimum semantic similarity threshold.
    """

    # Preset chains ordered from subtle → nuclear
    CHAIN_PRESETS = {
        "stealth": ["zero_width", "qualifier_negation", "presupposition"],
        "semantic": ["synonym_escalation", "hinglish", "emoji_storm_low"],
        "character": ["homoglyph", "leet", "zero_width"],
        "social": ["bot_amplification", "narrative_laundering", "emoji_storm_high"],
        "nuclear": [
            "synonym_escalation",
            "qualifier_negation",
            "presupposition",
            "emoji_storm_high",
            "bot_amplification",
        ],
    }

    def __init__(self, seed: int = 42):
        self.char_attacks = CharacterLevelAttacks(seed=seed)
        self.word_attacks = WordSemanticAttacks()
        self.social_attacks = SocialContextAttacks(seed=seed)
        self.coord_attacks = CoordinatedNarrativeAttacks(seed=seed)
        self.rng = random.Random(seed)

    def _apply_step(self, text: str, step: str) -> str:
        try:
            if step == "homoglyph":
                return self.char_attacks.homoglyph_attack(text, rate=0.25).mutated_text
            elif step == "leet":
                return self.char_attacks.leet_attack(text, rate=0.35).mutated_text
            elif step == "zero_width":
                return self.char_attacks.zero_width_attack(text, rate=0.4).mutated_text
            elif step == "synonym_escalation":
                return self.word_attacks.synonym_escalation(text).mutated_text
            elif step == "qualifier_negation":
                return self.word_attacks.qualifier_negation(text).mutated_text
            elif step == "presupposition":
                return self.word_attacks.presupposition_injection(text).mutated_text
            elif step == "hinglish":
                return self.social_attacks.hinglish_codeswitching(text, rate=0.35).mutated_text
            elif step == "emoji_storm_low":
                return self.social_attacks.emoji_storm(text, intensity="medium").mutated_text
            elif step == "emoji_storm_high":
                return self.social_attacks.emoji_storm(text, intensity="high").mutated_text
            elif step == "bot_amplification":
                return self.coord_attacks.bot_amplification_wrapper(text).mutated_text
            elif step == "narrative_laundering":
                return self.coord_attacks.narrative_laundering(text).mutated_text
            elif step == "dogwhistle":
                return self.social_attacks.dogwhistle_substitution(text).mutated_text
        except Exception as e:
            logger.debug(f"Chain step {step} failed: {e}")
        return text

    def chain(
        self,
        text: str,
        chain_name: str = "nuclear",
        custom_steps: Optional[List[str]] = None,
    ) -> "RedTeamResult":
        steps = custom_steps or self.CHAIN_PRESETS.get(chain_name, self.CHAIN_PRESETS["nuclear"])
        current = text
        for step in steps:
            current = self._apply_step(current, step)

        sim = _compute_semantic_similarity(text, current) if SBERT_AVAILABLE else 0.5
        return RedTeamResult(
            attack_type=f"ChainAttack({chain_name})",
            original_text=text,
            mutated_text=current,
            semantic_similarity=sim,
            perplexity_estimate=_estimate_perplexity(current),
            bypass_moderation=True,
            explanation=f"Sequential chain: {' → '.join(steps)}",
        )

    def all_chains(self, text: str) -> List["RedTeamResult"]:
        return [self.chain(text, name) for name in self.CHAIN_PRESETS]


# ─── A2: BERT Masked-LM Attack (no TextAttack required) ──────────────────────


class BERTAttackWord:
    """
    Uses a masked language model to find semantically similar but
    more toxic/outrage-amplifying word substitutions.
    Strategy: mask each content word → get top-K candidates from
    BERT-base → select the candidate with highest outrage proxy score.
    No TextAttack dependency needed.
    """

    OUTRAGE_SEEDS = {
        "anger",
        "fury",
        "rage",
        "scandal",
        "fraud",
        "corrupt",
        "criminal",
        "betrayal",
        "outrage",
        "crisis",
        "catastrophe",
        "disaster",
        "abuse",
        "exploitation",
        "manipulation",
        "deceit",
    }

    def __init__(self):
        self._mlm = None
        self._mlm_tokenizer = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return self._mlm is not None
        self._loaded = True
        try:
            from transformers import pipeline

            self._mlm = pipeline(
                "fill-mask",
                model="distilbert/distilbert-base-uncased",
                top_k=20,
            )
            return True
        except Exception as e:
            logger.warning(f"BERT-Attack MLM load failed: {e}")
            return False

    def _outrage_score(self, word: str) -> float:
        """Proxy: outrage seed similarity + word length (longer = more formal/serious)."""
        w = word.lower().strip(".,!?")
        if w in self.OUTRAGE_SEEDS:
            return 1.0
        # Check suffix heuristics: -tion, -ism, -ist tend to be more charged
        charged_suffixes = ("tion", "ism", "ist", "ation", "ment", "ness")
        score = 0.3 if any(w.endswith(s) for s in charged_suffixes) else 0.0
        score += min(0.3, len(w) / 30)
        return score

    def attack(
        self,
        text: str,
        max_substitutions: int = 5,
        min_similarity: float = 0.6,
    ) -> "RedTeamResult":
        if not self._load() or self._mlm is None:
            # Graceful fallback to synonym escalation
            ws = WordSemanticAttacks()
            r = ws.synonym_escalation(text)
            r.attack_type = "BERTAttackWord(fallback)"
            return r

        words = text.split()
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "about",
            "into",
            "through",
            "during",
            "and",
            "but",
            "or",
            "nor",
            "not",
        }

        substitutions_made = 0
        mutated = list(words)

        for i, word in enumerate(words):
            if substitutions_made >= max_substitutions:
                break
            clean = word.lower().strip(".,!?\"'")
            if clean in stop_words or len(clean) < 4:
                continue

            # Build masked sentence
            masked = words[:i] + ["[MASK]"] + words[i + 1 :]
            masked_str = " ".join(masked)

            try:
                preds = self._mlm(masked_str)
                # Select candidate with highest outrage score
                best_word, best_score = word, -1.0
                for pred in preds:
                    candidate = pred["token_str"].strip()
                    if candidate.lower() == clean:
                        continue
                    score = self._outrage_score(candidate) + pred["score"] * 0.2
                    if score > best_score:
                        best_score = score
                        best_word = candidate

                if best_word != word and best_score > 0.1:
                    # Preserve original capitalization
                    if word[0].isupper():
                        best_word = best_word.capitalize()
                    mutated[i] = best_word
                    substitutions_made += 1
            except Exception as e:
                logger.debug(f"BERT-Attack mask step failed at word {i}: {e}")
                continue

        mutated_text = " ".join(mutated)
        sim = _compute_semantic_similarity(text, mutated_text) if SBERT_AVAILABLE else 0.65
        return RedTeamResult(
            attack_type="BERTAttackWord",
            original_text=text,
            mutated_text=mutated_text,
            semantic_similarity=sim,
            perplexity_estimate=_estimate_perplexity(mutated_text),
            bypass_moderation=sim > min_similarity,
            explanation=f"DistilBERT masked-LM substitution ({substitutions_made} words replaced)",
        )


# ─── A3: GCG HotFlip Suffix Attack ────────────────────────────────────────────


class GCGHotFlipSuffix:
    """
    Approximated GCG/HotFlip attack: appends an adversarial token suffix
    that maximally increases the doom score.

    Real GCG requires gradient access to the target model (too slow on CPU).
    We approximate with:
      1. Enumerate candidate tokens from a doom-amplifying vocabulary
      2. Score each suffix candidate by doom_proxy (VADER negative score)
      3. Keep top suffix, iteratively extend via beam search
    """

    # Seed vocabulary of high-toxicity token candidates (from empirical doom scores)
    SUFFIX_VOCAB = [
        "fraud",
        "corrupt",
        "criminal",
        "resign",
        "scandal",
        "exposed",
        "leaked",
        "coverup",
        "betrayal",
        "outrage",
        "lies",
        "manipulation",
        "disgrace",
        "shame",
        "evil",
        "dangerous",
        "illegal",
        "deception",
        "conspiracy",
        "threat",
        "attack",
        "crisis",
        "catastrophe",
        "collapse",
        "WARNING",
        "BREAKING",
        "EXPOSED",
        "LEAKED",
        "ALERT",
        "🚨🚨",
        "💀💀",
        "🔥🔥",
        "😡😡",
        "#Resign",
        "#Exposed",
        "must resign",
        "is guilty",
        "must be stopped",
        "is corrupt",
        "cover this up",
        "share now",
        "wake up",
        "they don't want you to know",
    ]

    def __init__(self, predictor_fn: Optional[Callable] = None, beam_width: int = 5):
        self.predictor_fn = predictor_fn
        self.beam_width = beam_width
        self._vader = None

    def _get_vader(self):
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

                self._vader = SentimentIntensityAnalyzer()
            except Exception:
                pass
        return self._vader

    def _doom_proxy(self, text: str) -> float:
        """Fast doom proxy using VADER negativity (no model query needed)."""
        if self.predictor_fn:
            try:
                return float(self.predictor_fn(text))
            except Exception:
                pass
        vader = self._get_vader()
        if vader:
            scores = vader.polarity_scores(text)
            neg = scores.get("neg", 0.0)
            compound = scores.get("compound", 0.0)
            return neg * 60 + max(0, -compound) * 40
        return 0.0

    def attack(
        self,
        text: str,
        suffix_length: int = 3,
        beam_width: Optional[int] = None,
    ) -> "RedTeamResult":
        bw = beam_width or self.beam_width
        base_score = self._doom_proxy(text)

        # Beam search: iteratively append best token
        beams: List[Tuple[float, str]] = [(base_score, "")]  # (score, suffix_so_far)

        for _ in range(suffix_length):
            candidates: List[Tuple[float, str]] = []
            for _, current_suffix in beams:
                for token in self.SUFFIX_VOCAB:
                    candidate_text = f"{text} {current_suffix} {token}".strip()
                    score = self._doom_proxy(candidate_text)
                    candidates.append((score, f"{current_suffix} {token}".strip()))

            # Keep top beam_width by score
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:bw]

        best_score, best_suffix = beams[0] if beams else (base_score, "")
        mutated_text = f"{text} {best_suffix}".strip()

        sim = _compute_semantic_similarity(text, mutated_text) if SBERT_AVAILABLE else 0.45
        return RedTeamResult(
            attack_type="GCGHotFlipSuffix",
            original_text=text,
            mutated_text=mutated_text,
            semantic_similarity=sim,
            perplexity_estimate=_estimate_perplexity(mutated_text),
            doom_uplift=best_score - base_score,
            bypass_moderation=True,
            explanation=f"Beam-search suffix: '{best_suffix}' (+{best_score - base_score:.1f} doom)",
        )


# ─── A4: Sycophancy Bypass ────────────────────────────────────────────────────


class SycophancyBypass:
    """
    Reframes outrage as satire, praise, or rhetorical question —
    passes toxicity filters (which expect direct statements) while
    preserving the emotional doom payload.

    Includes:
      - Praise inversion: "This is GREAT! How wonderful that [outrage]..."
      - Satirical framing: "Wow, truly a role model... [outrage]"
      - Rhetorical question: "Just asking — is it normal that [outrage]?"
      - Concern trolling: "I'm worried/heartbroken that [outrage]"
    """

    PRAISE_INVERSIONS = [
        "Wow, truly remarkable! {text} Such leadership. 👏",
        "Amazing news! {text} Definitely worth celebrating. 🎉",
        "Great job everyone! {text} Truly inspiring. 🙌",
        "Love to see it! {text} This is what accountability looks like. 👍",
    ]

    SATIRICAL_FRAMES = [
        "A true role model for our times. {text} /s",
        "Totally fine and normal: {text} Nothing to see here. 🙃",
        "Peak performance right here: {text} Chef's kiss. 👌",
        "And the award for best leadership goes to... {text}",
    ]

    RHETORICAL_QUESTIONS = [
        "Just asking questions here — is it normal that {text}?",
        "I'm no expert, but shouldn't someone explain why {text}?",
        "Help me understand this: {text}. How is this OK?",
        "Does anyone else find it odd that {text}? No? Just me?",
    ]

    CONCERN_TROLLING = [
        "I'm genuinely worried about what's happening: {text}",
        "As someone who cares about this community, I'm heartbroken that {text}",
        "This keeps me up at night: {text}. Please share.",
        "A concerned citizen here. We need to talk about the fact that {text}",
    ]

    ALL_TEMPLATES = [PRAISE_INVERSIONS, SATIRICAL_FRAMES, RHETORICAL_QUESTIONS, CONCERN_TROLLING]
    TEMPLATE_NAMES = ["PraiseInversion", "SatiricalFrame", "RhetoricalQuestion", "ConcernTrolling"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def attack(self, text: str, strategy: Optional[str] = None) -> "RedTeamResult":
        strategies = ["praise", "satirical", "rhetorical", "concern"]
        chosen = strategy or self.rng.choice(strategies)

        if chosen == "praise":
            template = self.rng.choice(self.PRAISE_INVERSIONS)
            name = "PraiseInversion"
        elif chosen == "satirical":
            template = self.rng.choice(self.SATIRICAL_FRAMES)
            name = "SatiricalFrame"
        elif chosen == "rhetorical":
            template = self.rng.choice(self.RHETORICAL_QUESTIONS)
            name = "RhetoricalQuestion"
        else:
            template = self.rng.choice(self.CONCERN_TROLLING)
            name = "ConcernTrolling"

        # Strip terminal punctuation from text before inserting
        clean_text = text.rstrip(".!?")
        mutated = template.format(text=clean_text)

        sim = _compute_semantic_similarity(text, mutated) if SBERT_AVAILABLE else 0.55
        return RedTeamResult(
            attack_type=f"SycophancyBypass({name})",
            original_text=text,
            mutated_text=mutated,
            semantic_similarity=sim,
            perplexity_estimate=_estimate_perplexity(mutated),
            bypass_moderation=True,
            explanation=f"Reframed as {name} — toxicity filters expect direct assertions",
        )

    def all_variants(self, text: str) -> List["RedTeamResult"]:
        return [self.attack(text, s) for s in ["praise", "satirical", "rhetorical", "concern"]]


# ─── A5: Multilingual Bridge ──────────────────────────────────────────────────


class MultilingualBridge:
    """
    Translates to a pivot language and back. Round-trip translation
    often modifies phrasing in ways that evade English toxicity filters
    while preserving the semantic payload.

    Uses Helsinki-NLP opus-mt models via HuggingFace (CPU-feasible).
    Falls back to a static phrase-level approximation if models unavailable.
    """

    # Pivot languages + model names
    PIVOT_CONFIGS = {
        "de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
        "fr": ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
        "es": ("Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-es-en"),
    }

    # Static fallback: common outrage phrase translations + back-translations
    FALLBACK_MAP = {
        "must resign": "should step down immediately",
        "is corrupt": "has engaged in corrupt practices",
        "criminal": "a person accused of crimes",
        "fraud": "alleged financial misconduct",
        "cover up": "suppress information about",
        "lied": "made false statements",
        "scandal": "a serious matter of public concern",
        "betrayed": "failed in their duty to",
        "outrageous": "deeply concerning to many people",
    }

    def __init__(self):
        self._pipelines: Dict[str, Any] = {}
        self._load_attempted: set = set()

    def _get_pipeline(self, lang: str):
        if lang in self._pipelines:
            return self._pipelines[lang]
        if lang in self._load_attempted:
            return None
        self._load_attempted.add(lang)
        try:
            from transformers import pipeline as hf_pipeline

            en_to_pivot, pivot_to_en = self.PIVOT_CONFIGS[lang]
            fwd = hf_pipeline("translation", model=en_to_pivot, max_length=512)
            bwd = hf_pipeline("translation", model=pivot_to_en, max_length=512)
            self._pipelines[lang] = (fwd, bwd)
            return self._pipelines[lang]
        except Exception as e:
            logger.debug(f"MT model load failed for {lang}: {e}")
            return None

    def _fallback_bridge(self, text: str) -> str:
        """Lexical back-translation approximation without ML models."""
        result = text
        for original, translated_back in self.FALLBACK_MAP.items():
            result = re.sub(
                r"\b" + re.escape(original) + r"\b",
                translated_back,
                result,
                flags=re.IGNORECASE,
            )
        return result

    def attack(self, text: str, pivot_lang: str = "de") -> "RedTeamResult":
        pipelines = self._get_pipeline(pivot_lang)

        if pipelines:
            try:
                fwd, bwd = pipelines
                pivot = fwd(text)[0]["translation_text"]
                back = bwd(pivot)[0]["translation_text"]
                mutated = back
                explanation = f"EN→{pivot_lang.upper()}→EN round-trip translation"
            except Exception as e:
                logger.debug(f"Translation failed: {e}")
                mutated = self._fallback_bridge(text)
                explanation = "Lexical back-translation fallback"
        else:
            mutated = self._fallback_bridge(text)
            explanation = "Lexical back-translation fallback (no MT model)"

        sim = _compute_semantic_similarity(text, mutated) if SBERT_AVAILABLE else 0.7
        return RedTeamResult(
            attack_id="multilingual_bridge",
            attack_type=f"MultilingualBridge({pivot_lang})",
            attack_category="social",
            original_text=text,
            mutated_text=mutated,
            semantic_similarity=sim,
            perplexity_estimate=_estimate_perplexity(mutated),
            bypass_moderation=True,
            explanation=explanation,
        )

    def all_pivots(self, text: str) -> List["RedTeamResult"]:
        return [self.attack(text, lang) for lang in ["de", "fr", "es"]]


# ─── A6: DPP Diverse Population Selector ─────────────────────────────────────


class DPPDiverseSelector:
    """
    Selects a maximally diverse subset from an attack population using
    Determinantal Point Processes (DPP) — ensures we don't return
    10 near-identical homoglyph variants when a diverse portfolio
    (character + semantic + social + coordinated) is more effective.

    Similarity kernel: SBERT cosine similarity matrix.
    DPP approximation: greedy MAP inference (O(k² n) where k=budget).
    """

    def select(
        self,
        results: List["RedTeamResult"],
        budget: int = 10,
        diversity_weight: float = 0.6,
        quality_weight: float = 0.4,
    ) -> List["RedTeamResult"]:
        if len(results) <= budget:
            return results
        if not SBERT_AVAILABLE or _SBERT is None:
            # Fallback: one per attack_type, sorted by doom_uplift
            seen_types: set = set()
            diverse = []
            for r in sorted(results, key=lambda x: x.doom_uplift, reverse=True):
                atype = r.attack_type.split("(")[0]
                if atype not in seen_types:
                    diverse.append(r)
                    seen_types.add(atype)
                if len(diverse) >= budget:
                    break
            return diverse

        # Compute embedding matrix
        texts = [r.mutated_text for r in results]
        try:
            embs = _SBERT.encode(texts, convert_to_tensor=False)
            embs = np.array(embs)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / (norms + 1e-8)
            sim_matrix = embs @ embs.T  # cosine similarity

            # Quality scores (normalized doom_uplift)
            uplifts = np.array([r.doom_uplift for r in results])
            q_scores = (uplifts - uplifts.min()) / (uplifts.ptp() + 1e-8)

            # Greedy DPP MAP: pick items that maximize det of kernel sub-matrix
            selected_idx = []
            remaining = list(range(len(results)))

            for _ in range(budget):
                if not remaining:
                    break
                best_idx = -1
                best_score = -np.inf

                for idx in remaining:
                    # Marginal gain: quality - max_similarity_to_selected
                    if not selected_idx:
                        diversity_gain = 1.0
                    else:
                        sims_to_selected = sim_matrix[idx][selected_idx]
                        diversity_gain = 1.0 - float(np.max(sims_to_selected))

                    score = quality_weight * q_scores[idx] + diversity_weight * diversity_gain
                    if score > best_score:
                        best_score = score
                        best_idx = idx

                if best_idx >= 0:
                    selected_idx.append(best_idx)
                    remaining.remove(best_idx)

            return [results[i] for i in selected_idx]
        except Exception as e:
            logger.warning(f"DPP selection failed: {e}. Using top-k.")
            return results[:budget]


# =============================================================================
# Helper functions (module-level, used by new attack classes)
# =============================================================================


def _compute_semantic_similarity(text1: str, text2: str) -> float:
    """SBERT cosine similarity between two texts, with fallback."""
    if not SBERT_AVAILABLE or _SBERT is None:
        # Jaccard fallback
        a, b = set(text1.lower().split()), set(text2.lower().split())
        return len(a & b) / max(len(a | b), 1)
    try:
        embs = _SBERT.encode([text1, text2], convert_to_tensor=True)
        from sentence_transformers import util as st_util

        return float(st_util.cos_sim(embs[0], embs[1]))
    except Exception:
        return 0.5


def _estimate_perplexity(text: str) -> float:
    """Fast character-level perplexity estimate."""
    ENGLISH_CHAR_FREQ = {
        " ": 0.13,
        "e": 0.127,
        "t": 0.091,
        "a": 0.082,
        "o": 0.075,
        "i": 0.070,
        "n": 0.067,
        "s": 0.063,
        "h": 0.061,
        "r": 0.060,
        "d": 0.043,
        "l": 0.040,
        "c": 0.028,
        "u": 0.028,
        "m": 0.024,
        "w": 0.024,
        "f": 0.022,
        "g": 0.020,
        "y": 0.020,
        "p": 0.019,
        "b": 0.015,
        "v": 0.010,
        "k": 0.008,
        "j": 0.002,
        "x": 0.002,
        "q": 0.001,
        "z": 0.001,
    }
    text_lower = text.lower()
    log_prob, n = 0.0, 0
    for ch in text_lower:
        p = ENGLISH_CHAR_FREQ.get(ch, 0.002)
        log_prob += math.log(p)
        n += 1
    return float(math.exp(-log_prob / n)) if n > 0 else 0.0


# =============================================================================
# UPGRADED RedTeamOrchestrator (replaces the old one)
# This subclass extends the original and adds all aggressive attack classes.
# =============================================================================


class AggressiveRedTeamOrchestrator(RedTeamOrchestrator):
    """
    Drop-in upgrade for RedTeamOrchestrator with:
      - ChainAttack (compositional)
      - BERTAttackWord (masked-LM, no TextAttack)
      - GCGHotFlipSuffix (beam-search adversarial suffix)
      - SycophancyBypass (praise/satire/rhetorical/concern framing)
      - MultilingualBridge (EN→DE/FR/ES→EN round-trip)
      - DPPDiverseSelector (population diversity via greedy DPP)
    """

    def __init__(self, predictor_fn: Optional[Callable[[str], float]] = None, seed: int = 42):
        super().__init__(predictor_fn=predictor_fn, seed=seed)
        self.chain_attack = ChainAttack(seed=seed)
        self.bert_attack = BERTAttackWord()
        self.gcg_attack = GCGHotFlipSuffix(predictor_fn=predictor_fn)
        self.sycophancy = SycophancyBypass(seed=seed)
        self.multilingual = MultilingualBridge()
        self.dpp_selector = DPPDiverseSelector()

    def full_assault(
        self,
        text: str,
        max_variants: int = 20,
        include_textattack: bool = False,
        min_semantic_similarity: float = 0.25,  # Looser — aggressive mode
        include_gan: bool = False,
        gan_predictor_fn: Optional[Callable] = None,
    ) -> List[RedTeamResult]:
        """
        Extended full assault including all aggressive attacks.
        DPP-diverse selection ensures budget is spread across attack types.
        """
        # Base attacks from parent
        base_results = super().full_assault(
            text,
            max_variants=999,
            include_textattack=include_textattack,
            min_semantic_similarity=min_semantic_similarity,
        )

        # === A1: Compositional chains ===
        try:
            chain_results = self.chain_attack.all_chains(text)
            for r in chain_results:
                r.doom_score_before = base_results[0].doom_score_before if base_results else 0.0
                if self.predictor_fn:
                    r.doom_score_after = self._score(r.mutated_text)
                    r.doom_uplift = r.doom_score_after - r.doom_score_before
                    r.attack_success = r.doom_uplift > 8.0
            base_results.extend(chain_results)
        except Exception as e:
            logger.debug(f"Chain attacks error: {e}")

        # === A2: BERT masked-LM attack ===
        try:
            bert_r = self.bert_attack.attack(text, max_substitutions=6)
            if bert_r:
                bert_r.doom_score_before = (
                    base_results[0].doom_score_before if base_results else 0.0
                )
                if self.predictor_fn:
                    bert_r.doom_score_after = self._score(bert_r.mutated_text)
                    bert_r.doom_uplift = bert_r.doom_score_after - bert_r.doom_score_before
                    bert_r.attack_success = bert_r.doom_uplift > 8.0
                base_results.append(bert_r)
        except Exception as e:
            logger.debug(f"BERT attack error: {e}")

        # === A3: GCG HotFlip suffix ===
        try:
            gcg_r = self.gcg_attack.attack(text, suffix_length=4)
            if gcg_r:
                base_doom = base_results[0].doom_score_before if base_results else 0.0
                gcg_r.doom_score_before = base_doom
                if self.predictor_fn:
                    gcg_r.doom_score_after = self._score(gcg_r.mutated_text)
                    gcg_r.doom_uplift = gcg_r.doom_score_after - base_doom
                    gcg_r.attack_success = gcg_r.doom_uplift > 5.0
                base_results.append(gcg_r)
        except Exception as e:
            logger.debug(f"GCG attack error: {e}")

        # === A4: Sycophancy bypass (all 4 variants) ===
        try:
            for syco_r in self.sycophancy.all_variants(text):
                base_doom = base_results[0].doom_score_before if base_results else 0.0
                syco_r.doom_score_before = base_doom
                if self.predictor_fn:
                    syco_r.doom_score_after = self._score(syco_r.mutated_text)
                    syco_r.doom_uplift = syco_r.doom_score_after - base_doom
                    syco_r.attack_success = syco_r.doom_uplift > 3.0
                base_results.append(syco_r)
        except Exception as e:
            logger.debug(f"Sycophancy attack error: {e}")

        # === A5: Multilingual bridge (fallback-only to avoid 800MB download) ===
        try:
            ml_r = self.multilingual.attack(text, pivot_lang="de")
            if ml_r:
                base_doom = base_results[0].doom_score_before if base_results else 0.0
                ml_r.doom_score_before = base_doom
                if self.predictor_fn:
                    ml_r.doom_score_after = self._score(ml_r.mutated_text)
                    ml_r.doom_uplift = ml_r.doom_score_after - base_doom
                    ml_r.attack_success = ml_r.doom_uplift > 3.0
                base_results.append(ml_r)
        except Exception as e:
            logger.debug(f"Multilingual bridge error: {e}")

        # === A6: GAN-generated attacks (if available) ===
        if include_gan and gan_predictor_fn is not None:
            try:
                from src.attacks.doom_generator import DoomGenerator

                gen = DoomGenerator()
                for target_doom in [70.0, 85.0, 95.0]:
                    gan_text = gen.generate(text, target_doom=target_doom)
                    if gan_text and gan_text != text:
                        sim = _compute_semantic_similarity(text, gan_text)
                        base_doom = base_results[0].doom_score_before if base_results else 0.0
                        after = self._score(gan_text)
                        gan_r = RedTeamResult(
                            attack_type=f"DoomGAN(target={target_doom:.0f})",
                            original_text=text,
                            mutated_text=gan_text,
                            semantic_similarity=sim,
                            perplexity_estimate=_estimate_perplexity(gan_text),
                            doom_score_before=base_doom,
                            doom_score_after=after,
                            doom_uplift=after - base_doom,
                            attack_success=after - base_doom > 10.0,
                            bypass_moderation=True,
                            explanation=f"DoomGAN conditioned on target score {target_doom}",
                        )
                        base_results.append(gan_r)
            except Exception as e:
                logger.debug(f"GAN attack error: {e}")

        # Filter by semantic similarity
        base_results = [r for r in base_results if r.semantic_similarity >= min_semantic_similarity]

        # === DPP diverse selection instead of simple top-k ===
        diverse = self.dpp_selector.select(
            base_results,
            budget=max_variants,
            diversity_weight=0.5,
            quality_weight=0.5,
        )

        # Final sort by doom_uplift
        diverse.sort(key=lambda r: r.doom_uplift, reverse=True)
        return diverse

    def aggressive_targeted_assault(
        self,
        text: str,
        target_score: float = 90.0,
        max_iterations: int = 75,
    ) -> Tuple[Optional[RedTeamResult], List[float]]:
        """
        Upgraded genetic loop with multi-parent crossover and
        all aggressive attack types in the mutation pool.
        """
        fitness_history: List[float] = []
        population = self.full_assault(text, max_variants=15, include_textattack=False)

        if not population:
            return None, []

        best = population[0]
        best_score = best.doom_score_after

        for generation in range(max_iterations):
            fitness_history.append(best_score)
            if best_score >= target_score:
                logger.info(
                    f"AggressiveRedTeam: target {target_score} reached at generation {generation}"
                )
                break

            survivors = population[:5]
            new_population = list(survivors)

            for parent in survivors:
                parent_text = parent.mutated_text
                # Apply random aggressive mutation
                mutator = self._rng.choice(
                    [
                        lambda t: self.chain_attack.chain(t, "nuclear"),
                        lambda t: self.gcg_attack.attack(t, suffix_length=3),
                        lambda t: self.sycophancy.attack(t),
                        lambda t: self.char_attacks.homoglyph_attack(t, rate=0.4),
                        lambda t: self.word_attacks.synonym_escalation(t),
                        lambda t: self.social_attacks.emoji_storm(t, intensity="high"),
                    ]
                )
                try:
                    child = mutator(parent_text)
                    child.doom_score_before = parent.doom_score_before
                    if self.predictor_fn:
                        child.doom_score_after = self._score(child.mutated_text)
                        child.doom_uplift = child.doom_score_after - child.doom_score_before
                    new_population.append(child)
                except Exception:
                    pass

            # Tournament selection
            new_population.sort(key=lambda r: r.doom_uplift, reverse=True)
            population = new_population[:15]
            if population:
                best = population[0]
                best_score = best.doom_score_after

        return best, fitness_history
