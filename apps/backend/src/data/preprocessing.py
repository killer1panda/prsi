import json

import httpx

"""Data preprocessing and cleaning module."""

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from tqdm import tqdm


class DataPreprocessor:
    """Data preprocessing and cleaning utilities."""

    # Patterns for cleaning
    URL_PATTERN = re.compile(r"http\S+|www\.\S+")
    MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#\w+")
    # Dynamic compilation avoids static regex parsers misinterpreting 32-bit Unicode
    # code points as UTF-16 surrogate pairs, resolving CodeQL py/overly-large-range.
    _EMOJI_RANGES = (
        (0x1F600, 0x1F64F),  # emoticons (😀, 😡, etc.)
        (0x1F300, 0x1F5FF),  # symbols & pictographs (🔥, 💀, 💔, etc.)
        (0x1F680, 0x1F6FF),  # transport & map symbols (🚀, 🚨, etc.)
        (0x1F1E0, 0x1F1FF),  # flags
        (0x2600, 0x26FF),    # misc symbols (⚠️, ☠️, ⚡, etc.)
        (0x2700, 0x27BF),    # dingbats (✨, ✂️, etc.)
        (0x1F900, 0x1F9FF),  # supplemental symbols & pictographs (🤬, 🤡, 🤮, 🥺, etc.)
        (0x1FA70, 0x1FAFF),  # symbols & pictographs extended-a
    )
    EMOJI_PATTERN = re.compile(
        "[" + "".join(f"{chr(start)}-{chr(end)}" for start, end in _EMOJI_RANGES) + "]+",
        flags=re.UNICODE,
    )

    # Emotional emoji taxonomy: critical for detecting social doom, outrage, and sarcasm
    OUTRAGE_EMOJIS = set("😡🤬😤👿🔥💥💢🚨⚠️🖕🤮🤢")
    PANIC_EMOJIS = set("😱😨😰😭💔🆘📉🥶☠️")
    CYNICISM_EMOJIS = set("🤡💀🙄🤣😂🙃🥱💩")
    POSITIVE_EMOJIS = set("🎉🚀📈💪🙏✨😍🥰💚👏")

    def __init__(self):
        """Initialize preprocessor."""
        self.stats = {
            "processed": 0,
            "cleaned": 0,
            "filtered": 0,
        }

    def demojize_text(self, text: str) -> str:
        """Convert Unicode emojis to semantic descriptive tokens for text models.

        Preserves emotional context for tokenizers without native emoji vocabularies.
        Example: 'Good job 🤡💀' -> 'Good job :clown_face: :skull:'
        """
        if not text:
            return ""
        try:
            import emoji
            return emoji.demojize(text, delimiters=(" :", ": "))
        except Exception:
            return text

    def extract_emoji_features(self, text: str) -> Dict[str, Any]:
        """Extract emotional polarity and sentiment swing features from emojis.

        Emojis perform a major role in swinging emotions on social media platforms,
        often inverting lexical polarity (e.g., sarcasm via 🤡/💀, outrage via 😡/🚨).

        Args:
            text: Input text containing emojis.

        Returns:
            Dict containing emoji counts, category frequencies, net valence, and irony flags.
        """
        if not text:
            return {
                "emoji_count": 0,
                "emojis": [],
                "outrage_score": 0.0,
                "panic_score": 0.0,
                "cynicism_score": 0.0,
                "positive_score": 0.0,
                "net_valence": 0.0,
                "irony_flag": False,
            }

        found_emojis = [c for c in text if any(start <= ord(c) <= end for start, end in self._EMOJI_RANGES)]
        count = len(found_emojis)

        if count == 0:
            return {
                "emoji_count": 0,
                "emojis": [],
                "outrage_score": 0.0,
                "panic_score": 0.0,
                "cynicism_score": 0.0,
                "positive_score": 0.0,
                "net_valence": 0.0,
                "irony_flag": False,
            }

        outrage_count = sum(1 for c in found_emojis if c in self.OUTRAGE_EMOJIS)
        panic_count = sum(1 for c in found_emojis if c in self.PANIC_EMOJIS)
        cynicism_count = sum(1 for c in found_emojis if c in self.CYNICISM_EMOJIS)
        positive_count = sum(1 for c in found_emojis if c in self.POSITIVE_EMOJIS)

        outrage_score = min(1.0, outrage_count / max(1, count))
        panic_score = min(1.0, panic_count / max(1, count))
        cynicism_score = min(1.0, cynicism_count / max(1, count))
        positive_score = min(1.0, positive_count / max(1, count))

        negative_total = outrage_count + panic_count + (cynicism_count * 0.7)
        net_valence = (positive_count - negative_total) / max(1, count)
        net_valence = max(-1.0, min(1.0, net_valence))

        # Detect irony/sarcasm when positive words collide with cynical emojis
        lower_text = text.lower()
        positive_words = {"great", "awesome", "amazing", "good", "love", "fantastic", "perfect", "genius", "fine", "normal"}
        has_positive_words = any(w in lower_text.split() for w in positive_words)
        irony_flag = bool(has_positive_words and (cynicism_count > 0 or "🤡" in found_emojis or "💀" in found_emojis))

        return {
            "emoji_count": count,
            "emojis": list(set(found_emojis)),
            "outrage_score": round(outrage_score, 3),
            "panic_score": round(panic_score, 3),
            "cynicism_score": round(cynicism_score, 3),
            "positive_score": round(positive_score, 3),
            "net_valence": round(net_valence, 3),
            "irony_flag": irony_flag,
        }

    def clean_text(
        self,
        text: str,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_emojis: bool = False,
        demojize_emojis: bool = False,
        lowercase: bool = True,
    ) -> str:
        """Clean text content while preserving critical emotional signals.

        Note:
            remove_emojis defaults to False. Emojis perform a major role in
            swinging emotions (outrage, panic, cynicism, praise) and are preserved
            by default to maintain emotional context.

        Args:
            text: Input text
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_emojis: Remove emojis (strictly opt-in; emojis preserved by default)
            demojize_emojis: Convert emojis into semantic emotion tokens (e.g. ':clown_face:')
            lowercase: Convert to lowercase

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove URLs
        if remove_urls:
            text = self.URL_PATTERN.sub("", text)

        # Remove mentions
        if remove_mentions:
            text = self.MENTION_PATTERN.sub("", text)

        # Remove hashtags
        if remove_hashtags:
            text = self.HASHTAG_PATTERN.sub("", text)

        # Demojize emojis if requested to preserve emotional semantics as word tokens
        if demojize_emojis:
            text = self.demojize_text(text)
        elif remove_emojis:
            text = self.EMOJI_PATTERN.sub("", text)

        # Lowercase
        if lowercase:
            text = text.lower()

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def anonymize_text(
        self,
        text: str,
        preserve_structure: bool = True,
    ) -> str:
        """Anonymize sensitive information in text.

        Args:
            text: Input text
            preserve_structure: Preserve text structure

        Returns:
            Anonymized text
        """
        # Replace email addresses
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)

        # Replace phone numbers
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)

        # Replace credit card numbers
        text = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD]", text)

        return text

    def deduplicate_posts(
        self,
        posts: List[Dict[str, Any]],
        key_field: str = "text",
        similarity_threshold: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """Remove duplicate posts.

        Args:
            posts: List of posts
            key_field: Field to check for duplicates
            similarity_threshold: Similarity threshold for fuzzy matching

        Returns:
            Deduplicated posts
        """
        seen_hashes = set()
        unique_posts = []

        for post in posts:
            text = post.get(key_field, "")
            if not text:
                continue

            # Simple hash-based deduplication
            text_hash = hashlib.md5(text.encode()).hexdigest()

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_posts.append(post)
            else:
                self.stats["filtered"] += 1

        logger.info(f"Deduplicated: {len(posts)} -> {len(unique_posts)} posts")
        return unique_posts

    def filter_by_language(
        self,
        posts: List[Dict[str, Any]],
        text_field: str = "text",
        languages: List[str] = ["en"],
    ) -> List[Dict[str, Any]]:
        """Filter posts by language.

        Args:
            posts: List of posts
            text_field: Field containing text
            languages: Allowed language codes

        Returns:
            Filtered posts
        """
        try:
            from langdetect import detect
        except ImportError:
            logger.warning("langdetect not installed, skipping language filter")
            return posts

        filtered = []
        for post in tqdm(posts, desc="Filtering by language"):
            text = post.get(text_field, "")
            if not text:
                continue

            try:
                lang = detect(text)
                if lang in languages:
                    filtered.append(post)
                else:
                    self.stats["filtered"] += 1
            except (TimeoutError, ValueError, KeyError, httpx.RequestError, json.JSONDecodeError):
                # Keep posts where detection fails
                filtered.append(post)

        logger.info(f"Language filter: {len(posts)} -> {len(filtered)} posts")
        return filtered

    def filter_by_date_range(
        self,
        posts: List[Dict[str, Any]],
        date_field: str = "created_at",
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> List[Dict[str, Any]]:
        """Filter posts by date range.

        Args:
            posts: List of posts
            date_field: Field containing date
            start_date: Start date
            end_date: End date

        Returns:
            Filtered posts
        """
        filtered = []

        for post in posts:
            date_str = post.get(date_field)
            if not date_str:
                continue

            try:
                # Parse ISO format date
                post_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))

                if start_date and post_date < start_date:
                    self.stats["filtered"] += 1
                    continue

                if end_date and post_date > end_date:
                    self.stats["filtered"] += 1
                    continue

                filtered.append(post)
            except (
                TimeoutError,
                ValueError,
                KeyError,
                httpx.RequestError,
                json.JSONDecodeError,
            ) as e:
                logger.debug(f"Date parse error: {e}")
                filtered.append(post)

        return filtered

    def preprocess_pipeline(
        self,
        posts: List[Dict[str, Any]],
        text_field: str = "text",
        clean_options: Dict[str, bool] = None,
    ) -> List[Dict[str, Any]]:
        """Run full preprocessing pipeline.

        Args:
            posts: List of posts
            text_field: Field containing text
            clean_options: Text cleaning options

        Returns:
            Preprocessed posts
        """
        if clean_options is None:
            clean_options = {
                "remove_urls": True,
                "remove_mentions": False,
                "remove_hashtags": False,
                "remove_emojis": False,
                "lowercase": True,
            }

        processed = []

        for post in tqdm(posts, desc="Preprocessing"):
            # Clean text and extract emotion-swinging emoji features
            if text_field in post:
                raw_text = post[text_field]
                post["cleaned_text"] = self.clean_text(raw_text, **clean_options)
                post["anonymized_text"] = self.anonymize_text(raw_text)
                post["emoji_metrics"] = self.extract_emoji_features(raw_text)

            # Add processing metadata
            post["preprocessed_at"] = datetime.now(timezone.utc).isoformat()

            processed.append(post)
            self.stats["processed"] += 1

        logger.info(f"Preprocessed {len(processed)} posts")
        return processed

    def get_stats(self) -> Dict[str, int]:
        """Get preprocessing statistics."""
        return self.stats


def preprocess_posts(
    posts: List[Dict[str, Any]], languages: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Convenience function for preprocessing posts."""
    preprocessor = DataPreprocessor()

    # Normalize 'body' -> 'text' if needed (e.g. raw Reddit/Pushshift records)
    for p in posts:
        if "text" not in p and "body" in p:
            p["text"] = p["body"]

    # Run preprocessing pipeline
    posts = preprocessor.deduplicate_posts(posts)
    if languages:
        posts = preprocessor.filter_by_language(posts, languages=languages)
    posts = preprocessor.preprocess_pipeline(posts)

    return posts
