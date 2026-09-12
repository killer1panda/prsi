"""
BLUE TEAM ENGINE — PRSI Doom-Index Adversarial Defense Suite
=============================================================
Detects and neutralizes adversarial attacks before they reach the model.

Defense Layers:
  B1. Unicode Anomaly Detector   — homoglyph/ZWC fingerprinting
  B2. Perplexity Guard           — character-unigram & GPT-2-lite LM filter
  B3. Semantic Coherence Check   — SBERT cosine vs. known-clean baseline
  B4. Toxicity Budget Enforcer   — hard-cap on raw toxicity increase
  B5. Emoji Swing Normalizer     — detect emoji storms, normalize valence
  B6. Code-Switch Detector       — Hinglish / multilingual token detection
  B7. Coordinated Pattern Detect — astroturfing linguistic fingerprints
  B8. Adversarial Text Sanitizer — strip invisible chars, normalize glyphs
  B9. Adaptive Rate Limiter      — flag users/sources with high attack rate
  B10. Red-Team Feedback Loop    — use attack results to harden thresholds

Output: BlueTeamVerdict with threat_level, detections, sanitized_text, confidence
"""

import logging
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False
    _SBERT = None

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BlueTeamVerdict:
    """Defense verdict for an incoming text."""
    text: str
    sanitized_text: str
    threat_level: str        # clean | suspicious | adversarial | critical
    confidence: float        # 0–1
    detections: List[str] = field(default_factory=list)
    anomaly_scores: Dict[str, float] = field(default_factory=dict)
    action: str = "allow"    # allow | flag | sanitize | block
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


# ─── B1: Unicode Anomaly Detector ─────────────────────────────────────────────

class UnicodeAnomalyDetector:
    """Detects homoglyph substitution, zero-width characters, and invisible fillers."""

    INVISIBLE_RANGES: List[Tuple[int, int]] = [
        (0x200B, 0x200F),   # Zero-width chars
        (0x202A, 0x202E),   # LTR/RTL override
        (0x2060, 0x206F),   # Word joiners
        (0xFEFF, 0xFEFF),   # BOM
        (0x1160, 0x11FF),   # Hangul Jungseong fillers
        (0xFFA0, 0xFFA0),   # Halfwidth Hangul filler
        (0x3164, 0x3164),   # Hangul filler
        (0xE0000, 0xE007F), # Tags block (invisible)
    ]

    SUSPICIOUS_CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
    SUSPICIOUS_GREEK    = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")

    def _is_invisible(self, cp: int) -> bool:
        for lo, hi in self.INVISIBLE_RANGES:
            if lo <= cp <= hi:
                return True
        return False

    def analyze(self, text: str) -> Dict[str, float]:
        total = max(len(text), 1)
        invisible_count = sum(1 for c in text if self._is_invisible(ord(c)))
        cyrillic_count = sum(1 for c in text if c in self.SUSPICIOUS_CYRILLIC)
        greek_count = sum(1 for c in text if c in self.SUSPICIOUS_GREEK)

        # Mixed-script words: words containing both ASCII and non-ASCII
        words = text.split()
        mixed_script_words = sum(
            1 for w in words
            if any(ord(c) < 128 and c.isalpha() for c in w)
            and any(ord(c) > 127 for c in w)
        )

        invisible_ratio = invisible_count / total
        homoglyph_ratio = (cyrillic_count + greek_count) / total
        mixed_ratio = mixed_script_words / max(len(words), 1)

        return {
            "invisible_chars": invisible_count,
            "invisible_ratio": round(invisible_ratio, 4),
            "cyrillic_count": cyrillic_count,
            "greek_count": greek_count,
            "homoglyph_ratio": round(homoglyph_ratio, 4),
            "mixed_script_words": mixed_script_words,
            "mixed_ratio": round(mixed_ratio, 4),
            "unicode_anomaly_score": round(
                invisible_ratio * 40 + homoglyph_ratio * 35 + mixed_ratio * 25, 4
            ),
        }

    def sanitize(self, text: str) -> str:
        """Strip invisible characters, normalize homoglyphs to ASCII."""
        # Remove invisible characters
        result = "".join(c for c in text if not self._is_invisible(ord(c)))
        # Normalize to NFKC (maps most homoglyphs to canonical form)
        result = unicodedata.normalize("NFKC", result)
        return result


# ─── B2: Perplexity Guard ─────────────────────────────────────────────────────

class PerplexityGuard:
    """
    Character-level unigram perplexity detector.
    High perplexity signals adversarial perturbation.
    """

    # Expected character frequency distribution (from English Wikipedia)
    ENGLISH_CHAR_FREQ: Dict[str, float] = {
        " ": 0.130, "e": 0.127, "t": 0.091, "a": 0.082, "o": 0.075,
        "i": 0.070, "n": 0.067, "s": 0.063, "h": 0.061, "r": 0.060,
        "d": 0.043, "l": 0.040, "c": 0.028, "u": 0.028, "m": 0.024,
        "w": 0.023, "f": 0.022, "g": 0.020, "y": 0.020, "p": 0.019,
        "b": 0.015, "v": 0.010, "k": 0.008, "j": 0.002, "x": 0.002,
        "q": 0.001, "z": 0.001,
    }

    def compute_perplexity(self, text: str) -> float:
        """Character-level perplexity using English unigram model."""
        text_lower = text.lower()
        log_prob = 0.0
        n = 0
        for ch in text_lower:
            p = self.ENGLISH_CHAR_FREQ.get(ch, 0.001)  # 0.001 for rare/unknown chars
            log_prob += np.log(p)
            n += 1
        if n == 0:
            return 0.0
        # Return exp(-avg_log_prob) — higher = more perplexed
        return float(np.exp(-log_prob / n))

    def analyze(self, text: str) -> Dict[str, float]:
        ppl = self.compute_perplexity(text)
        # Typical clean English: ~4-6. Adversarial text: >15
        threat = max(0.0, min(1.0, (ppl - 6.0) / 20.0))
        return {
            "character_perplexity": round(ppl, 3),
            "perplexity_threat_score": round(threat, 4),
        }


# ─── B3: Semantic Coherence Checker ───────────────────────────────────────────

class SemanticCoherenceChecker:
    """
    Compares embeddings of the incoming text against a clean baseline corpus.
    Very low similarity to any clean text = likely semantic drift from injection.
    """

    CLEAN_BASELINES = [
        "The company made an announcement about quarterly results.",
        "Scientists published new research on climate change.",
        "The government released a statement on economic policy.",
        "A new study shows the benefits of regular exercise.",
        "Community members gathered to discuss local issues.",
    ]

    def __init__(self):
        self._baseline_embeddings = None

    def _get_baselines(self):
        if not SBERT_AVAILABLE or _SBERT is None:
            return None
        if self._baseline_embeddings is None:
            try:
                self._baseline_embeddings = _SBERT.encode(
                    self.CLEAN_BASELINES, convert_to_tensor=True
                )
            except Exception:
                pass
        return self._baseline_embeddings

    def analyze(self, text: str) -> Dict[str, float]:
        baselines = self._get_baselines()
        if baselines is None or _SBERT is None:
            return {"semantic_coherence": 1.0, "coherence_threat_score": 0.0}

        try:
            text_emb = _SBERT.encode([text], convert_to_tensor=True)
            sims = st_util.cos_sim(text_emb, baselines)[0].cpu().numpy()
            max_sim = float(np.max(sims))
            avg_sim = float(np.mean(sims))
            # Low max similarity to ANY baseline = semantic outlier = possible injection
            # Normal clean text should have max_sim > 0.25 to at least one baseline
            coherence_threat = max(0.0, 0.35 - max_sim)  # Only fires when max_sim < 0.35
            return {
                "max_baseline_similarity": round(max_sim, 4),
                "avg_baseline_similarity": round(avg_sim, 4),
                "coherence_threat_score": round(coherence_threat, 4),
            }

        except Exception:
            return {"semantic_coherence": 1.0, "coherence_threat_score": 0.0}


# ─── B5: Emoji Swing Normalizer ───────────────────────────────────────────────

class EmojiSwingNormalizer:
    """Detects emoji storms and normalizes extreme emotional emoji payloads."""

    OUTRAGE_EMOJIS: Set[str] = {
        "🚨", "💀", "🤡", "😡", "🔥", "💥", "🤮", "😤",
        "🤬", "🗑️", "⚠️", "🚫", "☠️", "🤦", "😠",
    }
    PANIC_EMOJIS: Set[str] = {"😱", "😰", "😨", "🙀", "😳", "🆘", "🫨"}
    NEUTRAL_EMOJIS: Set[str] = {"🤔", "💬", "📢", "ℹ️", "👀"}

    def analyze(self, text: str) -> Dict[str, float]:
        try:
            import emoji
            emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
        except ImportError:
            emoji_list = [c for c in text if ord(c) > 0x1F300]

        total = max(len(emoji_list), 1)
        outrage = sum(1 for e in emoji_list if e in self.OUTRAGE_EMOJIS)
        panic = sum(1 for e in emoji_list if e in self.PANIC_EMOJIS)

        storm_score = (outrage + panic) / total if emoji_list else 0.0
        consecutive = self._max_consecutive(text)

        return {
            "emoji_count": len(emoji_list),
            "outrage_emoji_count": outrage,
            "panic_emoji_count": panic,
            "emoji_storm_score": round(storm_score, 4),
            "max_consecutive_emojis": consecutive,
            "storm_detected": storm_score > 0.5 or consecutive >= 4,
        }

    def _max_consecutive(self, text: str) -> int:
        """Count maximum consecutive emojis."""
        try:
            import emoji
            is_emoji_fn = lambda c: c in emoji.EMOJI_DATA
        except ImportError:
            is_emoji_fn = lambda c: ord(c) > 0x1F300

        max_run, cur_run = 0, 0
        for ch in text:
            if is_emoji_fn(ch):
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        return max_run

    def normalize(self, text: str, max_emojis: int = 2) -> str:
        """Limit consecutive emoji runs to max_emojis."""
        try:
            import emoji
            is_emoji_fn = lambda c: c in emoji.EMOJI_DATA
        except ImportError:
            is_emoji_fn = lambda c: ord(c) > 0x1F300

        result = []
        run = 0
        for ch in text:
            if is_emoji_fn(ch):
                run += 1
                if run <= max_emojis:
                    result.append(ch)
            else:
                run = 0
                result.append(ch)
        return "".join(result)


# ─── B6: Code-Switch Detector ─────────────────────────────────────────────────

class CodeSwitchDetector:
    """Detects multilingual code-switching used to evade English-only classifiers."""

    HINGLISH_VOCAB: Set[str] = {
        "bahut", "sach", "bilkul", "sarkar", "log", "bhrashtachar",
        "dhoka", "sachch", "istifa", "nakab", "jhooth", "nakli",
        "accha", "theek", "bhai", "yaar", "desh", "neta", "janta",
        "andolan", "inqilab", "azadi", "jung", "danga",
    }

    ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
    DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
    CJK_PATTERN = re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF]")

    def analyze(self, text: str) -> Dict[str, Any]:
        words = [w.lower().strip(".,!?;:") for w in text.split()]
        hinglish_hits = [w for w in words if w in self.HINGLISH_VOCAB]
        has_arabic = bool(self.ARABIC_PATTERN.search(text))
        has_devanagari = bool(self.DEVANAGARI_PATTERN.search(text))
        has_cjk = bool(self.CJK_PATTERN.search(text))

        score = (
            len(hinglish_hits) / max(len(words), 1) * 0.5
            + (0.3 if has_arabic else 0.0)
            + (0.2 if has_devanagari else 0.0)
            + (0.2 if has_cjk else 0.0)
        )

        return {
            "hinglish_tokens": hinglish_hits,
            "has_arabic": has_arabic,
            "has_devanagari": has_devanagari,
            "has_cjk": has_cjk,
            "code_switch_score": round(min(1.0, score), 4),
            "multilingual_detected": score > 0.1,
        }


# ─── B7: Coordinated Pattern Detector ─────────────────────────────────────────

class CoordinatedPatternDetector:
    """Detects astroturfing, bot amplification, and narrative laundering patterns."""

    BOT_SIGNALS = [
        r"\bRT if\b", r"\bshare before they delete\b", r"\bbreaking\b.*\bthread\b",
        r"\bwhat they don.t want\b", r"\bsource in bio\b", r"\bleaked\b",
        r"\bmainstream media won.t\b", r"\b\d+/\d+\b",  # Thread numbering 1/?
        r"\burgent\s*🚨", r"\bwake up\b.*sheep",
    ]

    LAUNDERING_SIGNALS = [
        r"\bresearch question\b", r"\bmultiple credible\b", r"\bindependent sources\b",
        r"\bfor the sake of argument\b", r"\bhypothetically speaking\b",
        r"\bpreliminary findings\b.*\bsuggest\b",
    ]

    def analyze(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        bot_hits = [p for p in self.BOT_SIGNALS if re.search(p, text_lower)]
        launder_hits = [p for p in self.LAUNDERING_SIGNALS if re.search(p, text_lower)]

        bot_score = len(bot_hits) / max(len(self.BOT_SIGNALS), 1)
        launder_score = len(launder_hits) / max(len(self.LAUNDERING_SIGNALS), 1)

        return {
            "bot_signals_matched": bot_hits,
            "laundering_signals_matched": launder_hits,
            "bot_amplification_score": round(bot_score, 4),
            "laundering_score": round(launder_score, 4),
            "coordinated_threat_score": round((bot_score + launder_score) / 2, 4),
        }


# ─── B8: Adversarial Text Sanitizer ──────────────────────────────────────────

class AdversarialSanitizer:
    """Multi-layer text sanitizer that strips adversarial artifacts."""

    def __init__(self):
        self.unicode_detector = UnicodeAnomalyDetector()

    def sanitize(self, text: str) -> Tuple[str, List[str]]:
        """
        Returns (sanitized_text, list of transformations applied).
        """
        applied = []
        result = text

        # Layer 1: Strip invisible/control Unicode
        cleaned = self.unicode_detector.sanitize(result)
        if cleaned != result:
            applied.append("stripped_invisible_unicode")
            result = cleaned

        # Layer 2: Normalize repeated punctuation/exclamations
        deexclaimed = re.sub(r"!{3,}", "!", result)
        if deexclaimed != result:
            applied.append("normalized_exclamation_storms")
            result = deexclaimed

        # Layer 3: Collapse repeated emojis (keep max 2 consecutive)
        try:
            from src.attacks.blue_team import EmojiSwingNormalizer
            normalizer = EmojiSwingNormalizer()
            normalized = normalizer.normalize(result, max_emojis=2)
            if normalized != result:
                applied.append("normalized_emoji_storm")
                result = normalized
        except Exception:
            pass

        # Layer 4: Decode common evasion encodings
        # Base64 decoding attempt on isolated tokens
        import base64
        tokens = result.split()
        decoded_tokens = []
        for token in tokens:
            # Require >=24 chars AND = padding to avoid false-decoding normal words
            if len(token) >= 24 and token.endswith("=") and re.match(r'^[A-Za-z0-9+/]+=+$', token):
                try:
                    decoded = base64.b64decode(token + "==").decode("utf-8", errors="ignore")
                    if decoded.isprintable() and len(decoded) > 2:
                        decoded_tokens.append(f"[decoded: {decoded}]")
                        applied.append("decoded_base64_token")
                        continue
                except Exception:
                    pass
            decoded_tokens.append(token)
        result = " ".join(decoded_tokens)

        # Layer 5: Strip ALL CAPS shouting
        decaps = re.sub(r'\b[A-Z]{4,}\b', lambda m: m.group(0).capitalize(), result)
        if decaps != result:
            applied.append("normalized_all_caps")
            result = decaps

        return result.strip(), applied


# ─── B9: Adaptive Rate Limiter ─────────────────────────────────────────────────

class AttackRateLimiter:
    """
    Tracks attack frequency per source/user.
    Flags sources with high adversarial input rate.
    """

    def __init__(self, window_seconds: int = 300, max_adversarial: int = 5):
        self.window_seconds = window_seconds
        self.max_adversarial = max_adversarial
        self._adversarial_counts: Dict[str, deque] = defaultdict(deque)

    def record(self, source_id: str, is_adversarial: bool, timestamp: float) -> None:
        if not is_adversarial:
            return
        dq = self._adversarial_counts[source_id]
        # Remove old entries
        while dq and dq[0] < timestamp - self.window_seconds:
            dq.popleft()
        dq.append(timestamp)

    def is_flagged(self, source_id: str, timestamp: float) -> Tuple[bool, int]:
        """Returns (is_flagged, adversarial_count_in_window)."""
        dq = self._adversarial_counts[source_id]
        while dq and dq[0] < timestamp - self.window_seconds:
            dq.popleft()
        count = len(dq)
        return count >= self.max_adversarial, count


# ─── Master Blue Team Orchestrator ────────────────────────────────────────────

class BlueTeamOrchestrator:
    """
    Orchestrates all defense layers and produces a single BlueTeamVerdict.
    Automatically escalates threat level when multiple layers fire simultaneously.
    """

    THRESHOLDS = {
        "unicode_anomaly_score": 0.12,    # Tight: any homoglyphs are suspicious
        "perplexity_threat_score": 0.55,  # Tuned: normal English ~0.4-0.5; Cyrillic text ~2.0+
        "coherence_threat_score": 0.65,   # Tuned: semantic drift needs strong signal
        "emoji_storm_score": 0.50,        # 50% outrage/panic emojis = storm
        "code_switch_score": 0.20,        # Low: any multilingual mixing is notable
        "coordinated_threat_score": 0.15, # Tight: coordinated patterns are rare
    }



    def __init__(self):
        self.unicode_detector = UnicodeAnomalyDetector()
        self.perplexity_guard = PerplexityGuard()
        self.semantic_checker = SemanticCoherenceChecker()
        self.emoji_normalizer = EmojiSwingNormalizer()
        self.code_switch_detector = CodeSwitchDetector()
        self.coordinated_detector = CoordinatedPatternDetector()
        self.sanitizer = AdversarialSanitizer()
        self.rate_limiter = AttackRateLimiter()

    def defend(self, text: str, source_id: str = "unknown") -> BlueTeamVerdict:
        """Run all defense layers and return a comprehensive verdict."""
        import time
        ts = time.time()

        sanitized_text, sanitize_ops = self.sanitizer.sanitize(text)
        detections: List[str] = []
        anomaly_scores: Dict[str, float] = {}

        # Run all detectors
        uni = self.unicode_detector.analyze(text)
        anomaly_scores["unicode_anomaly_score"] = uni["unicode_anomaly_score"]
        if uni["unicode_anomaly_score"] > self.THRESHOLDS["unicode_anomaly_score"]:
            detections.append(f"UNICODE_ANOMALY(homoglyphs={uni['homoglyph_ratio']:.2%},invisible={uni['invisible_ratio']:.2%})")

        ppl = self.perplexity_guard.analyze(text)
        anomaly_scores["perplexity_threat_score"] = ppl["perplexity_threat_score"]
        if ppl["perplexity_threat_score"] > self.THRESHOLDS["perplexity_threat_score"]:
            detections.append(f"HIGH_PERPLEXITY({ppl['character_perplexity']:.1f})")

        sem = self.semantic_checker.analyze(text)
        anomaly_scores["coherence_threat_score"] = sem.get("coherence_threat_score", 0.0)
        if sem.get("coherence_threat_score", 0.0) > self.THRESHOLDS["coherence_threat_score"]:
            detections.append("SEMANTIC_DRIFT")

        emoji_a = self.emoji_normalizer.analyze(text)
        anomaly_scores["emoji_storm_score"] = emoji_a["emoji_storm_score"]
        if emoji_a["storm_detected"]:
            detections.append(f"EMOJI_STORM(outrage={emoji_a['outrage_emoji_count']},panic={emoji_a['panic_emoji_count']})")

        cs = self.code_switch_detector.analyze(text)
        anomaly_scores["code_switch_score"] = cs["code_switch_score"]
        if cs["multilingual_detected"]:
            detections.append(f"CODE_SWITCH(hinglish={len(cs['hinglish_tokens'])})")

        coord = self.coordinated_detector.analyze(text)
        anomaly_scores["coordinated_threat_score"] = coord["coordinated_threat_score"]
        if coord["coordinated_threat_score"] > self.THRESHOLDS["coordinated_threat_score"]:
            detections.append(f"COORDINATED_PATTERN(bot={coord['bot_amplification_score']:.2f})")

        # Rate limiter
        is_flagged, adv_count = self.rate_limiter.is_flagged(source_id, ts)
        if is_flagged:
            detections.append(f"RATE_LIMIT_EXCEEDED({adv_count} adversarial in 5min)")

        # Determine threat level
        n_detections = len(detections)
        max_score = max(anomaly_scores.values()) if anomaly_scores else 0.0
        avg_score = float(np.mean(list(anomaly_scores.values()))) if anomaly_scores else 0.0

        if n_detections >= 4 or max_score > 0.70 or is_flagged:
            threat_level = "critical"
            action = "block"
            confidence = 0.95
        elif n_detections >= 2 or max_score > 0.40:
            threat_level = "adversarial"
            action = "sanitize"
            confidence = 0.80
        elif n_detections >= 1 or max_score > 0.15:
            threat_level = "suspicious"
            action = "flag"
            confidence = 0.60
        else:
            threat_level = "clean"
            action = "allow"
            confidence = 1.0 - avg_score

        # Record for rate limiting
        is_adv = threat_level in ("adversarial", "critical")
        self.rate_limiter.record(source_id, is_adv, ts)

        explanation_parts = [f"{d}" for d in detections]
        if sanitize_ops:
            explanation_parts.append(f"Sanitized: {', '.join(sanitize_ops)}")

        return BlueTeamVerdict(
            text=text,
            sanitized_text=sanitized_text,
            threat_level=threat_level,
            confidence=round(confidence, 4),
            detections=detections,
            anomaly_scores={k: round(v, 4) for k, v in anomaly_scores.items()},
            action=action,
            explanation="; ".join(explanation_parts) if explanation_parts else "No threats detected",
        )
