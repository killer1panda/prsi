"""Data preprocessing and cleaning module."""

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from tqdm import tqdm


class DataPreprocessor:
    """Data preprocessing and cleaning utilities."""
    
    # Patterns for cleaning
    URL_PATTERN = re.compile(r'http\S+|www\.\S+')
    MENTION_PATTERN = re.compile(r'@\w+')
    HASHTAG_PATTERN = re.compile(r'#\w+')
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    
    def __init__(self):
        """Initialize preprocessor."""
        self.stats = {
            "processed": 0,
            "cleaned": 0,
            "filtered": 0,
        }
    
    def clean_text(
        self,
        text: str,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_emojis: bool = False,
        lowercase: bool = True,
    ) -> str:
        """Clean text content.
        
        Args:
            text: Input text
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_emojis: Remove emojis
            lowercase: Convert to lowercase
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        if remove_urls:
            text = self.URL_PATTERN.sub('', text)
        
        # Remove mentions
        if remove_mentions:
            text = self.MENTION_PATTERN.sub('', text)
        
        # Remove hashtags
        if remove_hashtags:
            text = self.HASHTAG_PATTERN.sub('', text)
        
        # Remove emojis
        if remove_emojis:
            text = self.EMOJI_PATTERN.sub('', text)
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
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
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            text
        )
        
        # Replace phone numbers
        text = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE]',
            text
        )
        
        # Replace credit card numbers
        text = re.sub(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            '[CARD]',
            text
        )
        
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
            except Exception:
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
                post_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                
                if start_date and post_date < start_date:
                    self.stats["filtered"] += 1
                    continue
                
                if end_date and post_date > end_date:
                    self.stats["filtered"] += 1
                    continue
                
                filtered.append(post)
            except Exception as e:
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
            # Clean text
            if text_field in post:
                post["cleaned_text"] = self.clean_text(
                    post[text_field],
                    **clean_options
                )
                post["anonymized_text"] = self.anonymize_text(post[text_field])
            
            # Add processing metadata
            post["preprocessed_at"] = datetime.utcnow().isoformat()
            
            processed.append(post)
            self.stats["processed"] += 1
        
        logger.info(f"Preprocessed {len(processed)} posts")
        return processed
    
    def get_stats(self) -> Dict[str, int]:
        """Get preprocessing statistics."""
        return self.stats


def preprocess_posts(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convenience function for preprocessing posts."""
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    posts = preprocessor.deduplicate_posts(posts)
    posts = preprocessor.filter_by_language(posts)
    posts = preprocessor.preprocess_pipeline(posts)
    
    return posts
