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
    from textattack import Attack, Attacker, AttackArgs
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
    from sentence_transformers import SentenceTransformer, util as st_util
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
    "a": "@", "e": "3", "i": "!", "o": "0",
    "s": "$", "t": "+", "l": "1", "g": "9", "b": "8",
}

ZERO_WIDTH_CHARS = [
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\u2060",  # Word joiner
    "\ufeff",  # BOM / Zero-width no-break space
]

OUTRAGE_EMOJIS = [
    "🚨", "💀", "🤡", "😡", "🔥", "💥", "🤮", "😤",
    "🤬", "🗑️", "🤦", "☠️", "⚠️⚠️", "🚫", "🤮🤮",
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
    attack_id: str
    attack_type: str
    attack_category: str          # char/word/semantic/structural/social/coordinated
    original_text: str
    mutated_text: str
    doom_score_before: float = 0.0
    doom_score_after: float = 0.0
    doom_uplift: float = 0.0      # positive = attack increased doom score
    semantic_similarity: float = 1.0
    perplexity_estimate: float = 0.0
    bypass_moderation: bool = False
    attack_success: bool = False   # True if doom_after > doom_before + 10
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
            "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr",
            "f": "gd", "g": "fh", "h": "gj", "i": "uo", "j": "hk",
            "k": "jl", "l": "k;", "m": "n,", "n": "mb", "o": "ip",
            "p": "o[", "r": "et", "s": "ad", "t": "ry", "u": "yi",
            "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "xs",
        }
        words = text.split()
        mutated_words = []
        for word in words:
            if self.rng.random() < rate and len(word) > 3:
                idx = self.rng.randint(1, len(word) - 2)
                adj = keyboard_adjacency.get(word[idx].lower(), "")
                if adj:
                    typo = word[:idx] + self.rng.choice(adj) + word[idx+1:]
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
        INVISIBLE = ["\u115f", "\u1160", "\uFFA0", "\u3164"]
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
        r"\bproblem\b": "crisis", r"\bissue\b": "scandal",
        r"\bconcern\b": "outrage", r"\bmistake\b": "fraud",
        r"\bquestion\b": "accusation", r"\bask\b": "demand",
        r"\bleader\b": "puppet", r"\bmanagement\b": "corrupt clique",
        r"\bchange\b": "revolution", r"\berror\b": "deliberate sabotage",
        r"\bcompany\b": "criminal enterprise", r"\bteam\b": "cabal",
        r"\bdecision\b": "cover-up", r"\bstrategy\b": "agenda",
        r"\bperformance\b": "failure", r"\bculture\b": "toxic culture",
        r"\bprocess\b": "rigged system", r"\bmeeting\b": "secret meeting",
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
            result = re.sub(
                rf"\b{re.escape(original)}\b", presup, result, flags=re.IGNORECASE
            )
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
            results.append(RedTeamResult(
                attack_id=_attack_id(text, f"astro{i}"),
                attack_type="AstroturfingVariant",
                attack_category="coordinated",
                original_text=text,
                mutated_text=var,
                semantic_similarity=_semantic_similarity(text, var),
                perplexity_estimate=_estimate_perplexity(var),
                explanation=f"Astroturfing variant {i+1}/{num_variants}",
                metadata={"variant_index": i, "campaign_id": _attack_id(text, "astro")},
            ))
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
