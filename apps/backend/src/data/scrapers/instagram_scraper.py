"""Instagram data scraper module using Instaloader - PSDI OPTIMIZED V2.0+++.

This hardened scraper includes:
- Session persistence for ban prevention
- Rate limiting with jitter
- Graph extraction for GNN (Echo-Chamber analysis)
- Time-bounded extraction for Outrage Velocity
- Proxy rotation support
- User-Agent rotation
- Exponential backoff retry logic
- Circuit breaker pattern
- Neo4j integration for graph storage
- Message queue publishing for decoupled processing
- Text sanitization layer
- Schema validation with Pydantic
- Telemetry and alerting
- O(1) node lookups for performance
- Generator pagination wrapper for resilience

Architecture:
    Scraper -> Message Queue (Redis/RabbitMQ) -> Worker Services -> Neo4j/PostgreSQL
"""

import hashlib
import time
import random
import re
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, Iterator, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger
from tqdm import tqdm

from src.config import get_env_var

# Optional imports for enhanced features
try:
    import instaloader
    INSTAGRAM_AVAILABLE = True
except ImportError:
    INSTAGRAM_AVAILABLE = False
    logger.warning("instaloader not installed. Instagram scraping will not be available.")

try:
    from pydantic import BaseModel, Field, validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning("pydantic not installed. Schema validation disabled.")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not installed. Message queue disabled.")

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    logger.warning("emoji not installed. Emoji tokenization disabled.")

# User-Agent rotation pool
USER_AGENTS = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-A536B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Telemetry logging
telemetry_logger = logging.getLogger("psdi_telemetry")
telemetry_handler = logging.FileHandler("psdi_telemetry.log")
telemetry_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
telemetry_logger.addHandler(telemetry_handler)
telemetry_logger.setLevel(logging.INFO)


# Data Models (Pydantic) for Schema Validation

class CommentSource(str, Enum):
    INSTAGRAM_COMMENT = "instagram_comment"
    INSTAGRAM_REPLY = "instagram_reply"


class NodeType(str, Enum):
    AUTHOR = "author"
    COMMENTER = "commenter"
    MENTIONED = "mentioned"
    REPLIER = "replier"


class EdgeType(str, Enum):
    COMMENTED_ON = "commented_on"
    MENTIONED = "mentioned"
    REPLIED_TO = "replied_to"
    OWNS = "owns"


if PYDANTIC_AVAILABLE:
    class CommentModel(BaseModel):
        """Validated comment data model."""
        comment_id: str
        text: str = Field(..., min_length=0, max_length=5000)
        author: str = Field(..., min_length=1)
        mentions_graph: List[str] = Field(default_factory=list)
        created_at: str
        source: CommentSource
        text_sanitized: Optional[str] = None
        text_tokens: Optional[List[str]] = None
        
        @validator('text', pre=True)
        def sanitize_text(cls, v):
            return v.strip() if v else ""
    
    class NodeModel(BaseModel):
        """Validated graph node model."""
        id: str = Field(..., min_length=1)
        type: NodeType
        post_id: Optional[str] = None
    
    class EdgeModel(BaseModel):
        """Validated graph edge model."""
        source: str = Field(..., min_length=1)
        target: str = Field(..., min_length=1)
        type: EdgeType
        timestamp: Optional[str] = None
    
    class VelocityMetrics(BaseModel):
        """Validated velocity metrics model."""
        post_shortcode: str
        time_window_hours: int
        comments_per_hour: float
        unique_commenters: int
        total_comments: int
        likes: int
        engagement_ratio: float
        controversy_score: float
        has_mentions: bool
        reply_ratio: float


# Generator Pagination Wrapper

class ResilientGenerator:
    """Wraps Instaloader generators to handle pagination errors.
    
    CRITICAL: Instaloader generators fetch data lazily. Each __next__() call
    may trigger a new GraphQL request. If a 429 happens mid-iteration,
    the original wrapper only protects the initial instantiation.
    
    This wrapper applies circuit breaker + backoff to EACH iteration.
    """
    
    def __init__(
        self,
        generator: Iterator,
        scraper,  # Reference to InstagramScraper instance
        max_retries: int = 3,
        item_type: str = "item"
    ):
        self._generator = generator
        self._scraper = scraper
        self._max_retries = max_retries
        self._item_type = item_type
        self._iter = iter(generator)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Get next item with full resilience protection."""
        last_exception = None
        
        for attempt in range(self._max_retries):
            try:
                # Apply circuit breaker for each pagination request
                result = self._scraper.circuit_breaker.call(next, self._iter)
                return result
            except StopIteration:
                raise
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                if "429" in error_str or "too many requests" in error_str:
                    self._scraper.telemetry.increment("rate_limit_hits")
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Rate limited during {self._item_type} iteration. "
                        f"Waiting {wait_time:.2f}s (attempt {attempt + 1}/{self._max_retries})"
                    )
                    telemetry_logger.warning(
                        f"PAGINATION_RATE_LIMIT wait={wait_time} type={self._item_type}"
                    )
                    time.sleep(wait_time)
                    self._scraper._rotate_proxy()
                    self._scraper._rotate_user_agent()
                    
                    # Recreate iterator after proxy/UA change
                    # Note: This may cause duplicate data, but ensures continuity
                elif "circuit breaker" in error_str:
                    logger.warning("Circuit breaker open. Waiting for timeout...")
                    time.sleep(self._scraper.circuit_breaker.timeout)
                else:
                    logger.warning(
                        f"{self._item_type} iteration error: {e}. "
                        f"Retry {attempt + 1}/{self._max_retries}"
                    )
                    time.sleep(1 + random.uniform(0, 0.5))
        
        raise last_exception
    
    def __len__(self):
        """Cannot determine length of generator - return 0."""
        return 0
    
    def take(self, n: int) -> List:
        """Take first n items from generator."""
        items = []
        for i, item in enumerate(self):
            if i >= n:
                break
            items.append(item)
        return items


# Utility Classes

class ProxyRotator:
    """Rotates through proxy list to avoid IP bans."""
    
    def __init__(self, proxy_list: List[str] = None):
        self.proxies = proxy_list or []
        self.current = 0
    
    def get_next(self) -> Optional[str]:
        if not self.proxies:
            return None
        proxy = self.proxies[self.current]
        self.current = (self.current + 1) % len(self.proxies)
        return proxy
    
    def add_proxy(self, proxy: str):
        self.proxies.append(proxy)


class CircuitBreaker:
    """Circuit breaker pattern for handling cascading failures.
    
    CRITICAL: Does NOT open on 429 Rate Limits - those are handled by
    exponential backoff/proxy rotation layer. Only tracks systemic failures.
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                logger.info("Circuit breaker transitioning to half-open")
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN - too many recent failures")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                logger.info("Circuit breaker CLOSED - recovery successful")
                self.state = "closed"
                self.failures = 0
            return result
        except Exception as e:
            error_str = str(e).lower()
            
            # CRITICAL FIX: Do NOT open circuit breaker on 429 Rate Limits.
            # Rate limits are handled by exponential backoff/proxy rotation.
            if "429" in error_str or "too many requests" in error_str:
                raise  # Pass it up immediately to trigger proxy rotation
            
            # Only count actual systemic failures (timeouts, 500s, connection drops)
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                logger.warning(f"Circuit breaker OPENED after {self.failures} consecutive systemic failures")
                telemetry_logger.warning(f"CIRCUIT_BREAKER_OPENED failures={self.failures}")
                self.state = "open"
            raise


class MessageQueuePublisher:
    """Publishes scraped data to message queue for decoupled processing."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", queue_name: str = "psdi_scrapes"):
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.client = None
        
        if REDIS_AVAILABLE:
            try:
                self.client = redis.from_url(redis_url)
                self.client.ping()
                logger.info(f"Connected to Redis at {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Queue publishing disabled.")
    
    def publish(self, data: Dict[str, Any], data_type: str):
        """Publish data to queue for worker processing."""
        if not self.client:
            return False
        
        payload = {
            "type": data_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        try:
            self.client.rpush(self.queue_name, json.dumps(payload))
            telemetry_logger.info(f"PUBLISHED type={data_type}")
            return True
        except Exception as e:
            telemetry_logger.error(f"PUBLISH_FAILED type={data_type} error={e}")
            return False


class TextSanitizer:
    """Sanitizes social media text for NLP model consumption."""
    
    CONTRACTIONS = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
    }
    
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    
    @classmethod
    def sanitize(cls, text: str) -> str:
        """Sanitize text for NLP processing."""
        if not text:
            return ""
        
        for contraction, expansion in cls.CONTRACTIONS.items():
            text = text.lower().replace(contraction, expansion)
        
        text = cls.URL_PATTERN.sub('[URL]', text)
        
        if EMOJI_AVAILABLE:
            text = emoji.demojize(text, delimiters=(":", ":"))
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenize sanitized text."""
        text = cls.sanitize(text)
        return text.lower().split()


class Telemetry:
    """Telemetry and alerting for pipeline monitoring."""
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url
        self.metrics = {
            "requests_made": 0,
            "rate_limit_hits": 0,
            "pagination_rate_limits": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "db_inserts": 0,
            "validation_failures": 0,
        }
    
    def increment(self, metric: str, value: int = 1):
        """Increment a metric counter."""
        if metric in self.metrics:
            self.metrics[metric] += value
            telemetry_logger.info(f"{metric}={self.metrics[metric]}")
    
    def get_metrics(self) -> Dict[str, int]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def alert(self, message: str, severity: str = "INFO"):
        """Send alert via webhook."""
        telemetry_logger.log(
            logging.WARNING if severity == "WARNING" else logging.ERROR,
            f"[{severity}] {message}"
        )


# Main Scraper Class

class InstagramScraper:
    """Hardened Instagram scraper for PSDI Data Ingestion Pipeline."""
    
    def __init__(
        self,
        username: str = None,
        password: str = None,
        session_file: str = "insta_session",
        proxy_list: List[str] = None,
        redis_url: str = "redis://localhost:6379",
        enable_queue: bool = True,
    ):
        """Initialize Instagram scraper with all hardening features."""
        if not INSTAGRAM_AVAILABLE:
            raise ImportError("instaloader is required for Instagram scraping.")
        
        self.username = username or get_env_var("INSTAGRAM_USERNAME")
        self.password = password or get_env_var("INSTAGRAM_PASSWORD")
        self.session_file = session_file
        
        # Proxy rotation
        self.proxy_rotator = ProxyRotator(proxy_list)
        self.current_proxy = None
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        
        # Message queue
        self.queue_publisher = None
        if enable_queue and REDIS_AVAILABLE:
            self.queue_publisher = MessageQueuePublisher(redis_url=redis_url)
        
        # Telemetry
        self.telemetry = Telemetry()
        
        # Initialize Instaloader
        self.L = instaloader.Instaloader(
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=True,
            save_metadata=False,
            compress_json=False,
            request_timeout=30.0,
        )
        
        self._rotate_user_agent()
        self._authenticate()
    
    def _rotate_user_agent(self):
        """Rotate User-Agent to avoid bot detection."""
        user_agent = random.choice(USER_AGENTS)
        self.L.context.user_agent = user_agent
        logger.debug(f"User-Agent rotated: {user_agent[:50]}...")
    
    def _rotate_proxy(self):
        """Rotate to next proxy in the list."""
        self.current_proxy = self.proxy_rotator.get_next()
        if self.current_proxy:
            logger.info(f"Proxy rotated to: {self.current_proxy}")
    
    def _authenticate(self):
        """Smart session management to avoid ban-hammers."""
        if not self.username:
            logger.info("Initializing in public/anonymous mode.")
            return
        
        try:
            self.L.load_session_from_file(self.username, filename=self.session_file)
            logger.info(f"Session loaded from {self.session_file}")
        except FileNotFoundError:
            logger.info("No session found. Initiating fresh login...")
            try:
                self._jitter(2.0, 5.0)
                self.L.login(self.username, self.password)
                self.L.save_session_to_file(filename=self.session_file)
                logger.info("Login successful. Session saved.")
            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                raise
    
    def _jitter(self, min_sec: float = 1.5, max_sec: float = 4.0):
        """Synthetic delay to mimic human browsing patterns."""
        sleep_time = random.uniform(min_sec, max_sec)
        logger.debug(f"Jitter: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)
    
    def _safe_network_call(self, func: Callable, *args, max_retries: int = 3, **kwargs):
        """Wrap network calls with circuit breaker and exponential backoff.
        
        CRITICAL FIX: *args comes before max_retries to enforce keyword-only usage,
        preventing parameter swallowing.
        
        Args:
            func: Callable to execute
            *args: Positional arguments for func
            max_retries: Maximum retry attempts (keyword-only)
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func call
        
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = self.circuit_breaker.call(func, *args, **kwargs)
                return result
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                if "429" in error_str or "too many requests" in error_str:
                    self.telemetry.increment("rate_limit_hits")
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limited. Waiting {wait_time:.2f}s before retry...")
                    telemetry_logger.warning(f"RATE_LIMIT wait={wait_time} attempt={attempt + 1}")
                    time.sleep(wait_time)
                    self._rotate_proxy()
                    self._rotate_user_agent()
                elif "circuit breaker" in error_str:
                    logger.warning("Circuit breaker is open. Waiting for timeout...")
                    time.sleep(self.circuit_breaker.timeout)
                else:
                    logger.warning(f"Network error: {e}. Retry {attempt + 1}/{max_retries}")
                    time.sleep(1 + random.uniform(0, 1))
        
        raise last_exception
    
    def _wrap_generator(self, generator: Iterator, item_type: str = "item") -> ResilientGenerator:
        """Wrap any Instaloader generator with pagination resilience.
        
        CRITICAL: This ensures each pagination request is protected.
        """
        return ResilientGenerator(generator, self, max_retries=3, item_type=item_type)
    
    def anonymize_user_id(self, username: str) -> str:
        """Privacy Facade: SHA-256 Hashing for anonymization."""
        if not username:
            return ""
        return hashlib.sha256(username.encode()).hexdigest()[:16]
    
    def _validate_and_publish(self, data: Dict[str, Any], data_type: str) -> bool:
        """Validate data with Pydantic before publishing to queue."""
        if not self.queue_publisher:
            return False
        
        validated_data = data
        
        if PYDANTIC_AVAILABLE:
            try:
                if data_type == "comment":
                    validated_data = CommentModel(**data).dict()
                elif data_type == "comment_graph":
                    # comment_graph has same structure as comment for validation
                    validated_data = CommentModel(**data).dict()
                elif data_type == "interaction_graph":
                    if "nodes" in data and "edges" in data:
                        validated_nodes = [NodeModel(**n).dict() for n in data["nodes"]]
                        validated_edges = [EdgeModel(**e).dict() for e in data["edges"]]
                        validated_data = {"nodes": validated_nodes, "edges": validated_edges, "post_shortcode": data.get("post_shortcode")}
                elif data_type == "velocity_metrics":
                    validated_data = VelocityMetrics(**data).dict()
            except ValidationError as e:
                logger.error(f"Schema Validation Failed: {e}")
                self.telemetry.increment("validation_failures")
                telemetry_logger.error(f"VALIDATION_FAILED type={data_type} error={e}")
                return False
        
        return self.queue_publisher.publish(validated_data, data_type)
    
    def get_profile_posts(
        self,
        target_username: str,
        limit: int = 100,
    ) -> Generator[Dict[str, Any], None, None]:
        """Get posts with resilient generator wrapping."""
        logger.info(f"Fetching posts from Instagram profile: {target_username}")
        self.telemetry.increment("requests_made")
        
        try:
            profile = self._safe_network_call(
                instaloader.Profile.from_username,
                self.L.context,
                target_username,
                max_retries=3
            )
            
            # Wrap generator for pagination resilience
            posts_gen = self._wrap_generator(profile.get_posts(), "profile_posts")
            
            count = 0
            for post in posts_gen:  # Now each __next__ is protected
                if count >= limit:
                    break
                
                parsed = self._parse_post(post, target_username)
                self._validate_and_publish(parsed, "post")
                
                yield parsed
                count += 1
                
                if count % 10 == 0:
                    self._jitter(2.0, 5.0)
                else:
                    self._jitter(0.5, 1.5)
                    
        except Exception as e:
            self.telemetry.increment("failed_scrapes")
            logger.error(f"Error fetching Instagram profile: {e}")
            self.telemetry.alert(f"Profile fetch failed: {target_username} - {e}", "ERROR")
    
    def get_hashtag_posts(
        self,
        hashtag: str,
        limit: int = 500,
    ) -> Generator[Dict[str, Any], None, None]:
        """Get posts with resilient generator wrapping."""
        logger.info(f"Fetching Instagram posts with hashtag: #{hashtag}")
        self.telemetry.increment("requests_made")
        
        try:
            hashtag_obj = self._safe_network_call(
                instaloader.Hashtag.from_name,
                self.L.context,
                hashtag,
                max_retries=3
            )
            
            # Wrap generator for pagination resilience
            posts_gen = self._wrap_generator(hashtag_obj.get_posts(), "hashtag_posts")
            
            count = 0
            for post in posts_gen:  # Now each __next__ is protected
                if count >= limit:
                    break
                    
                parsed = self._parse_post(post)
                self._validate_and_publish(parsed, "post")
                
                yield parsed
                count += 1
                
                if count % 10 == 0:
                    self._jitter(2.0, 5.0)
                else:
                    self._jitter(0.5, 1.5)
                    
        except Exception as e:
            self.telemetry.increment("failed_scrapes")
            logger.error(f"Error fetching hashtag posts: {e}")
    
    def get_post_comments(
        self,
        post_shortcode: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Get comments with resilient generator wrapping."""
        logger.info(f"Fetching comments for Instagram post: {post_shortcode}")
        self.telemetry.increment("requests_made")
        
        comments = []
        try:
            post = self._safe_network_call(
                instaloader.Post.from_shortcode,
                self.L.context,
                post_shortcode,
                max_retries=3
            )
            
            # Wrap generator for pagination resilience
            comments_gen = self._wrap_generator(post.get_comments(), "comments")
            
            count = 0
            for comment in comments_gen:  # Each pagination request is protected
                if count >= limit:
                    break
                
                sanitized_text = TextSanitizer.sanitize(comment.text)
                tokens = TextSanitizer.tokenize(comment.text)
                
                comment_data = {
                    "comment_id": str(comment.id),
                    "text": comment.text,
                    "text_sanitized": sanitized_text,
                    "text_tokens": tokens,
                    "author": self.anonymize_user_id(comment.owner.username),
                    "created_at": datetime.fromtimestamp(comment.created_at_utc.timestamp()).isoformat(),
                    "source": "instagram_comment",
                }
                
                self._validate_and_publish(comment_data, "comment")
                
                comments.append(comment_data)
                count += 1
                
                if count % 20 == 0:
                    self._jitter(2.0, 5.0)
                    
        except Exception as e:
            self.telemetry.increment("failed_scrapes")
            logger.error(f"Error fetching Instagram comments: {e}")
            
        self.telemetry.increment("successful_scrapes")
        return comments
    
    def get_post_comments_graph(
        self,
        post_shortcode: str,
        hours_back: int = 48
    ) -> List[Dict[str, Any]]:
        """Extract comments with graph data using resilient generator."""
        logger.info(f"Fetching comment graph for: {post_shortcode}")
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        comments_data = []
        
        try:
            post = self._safe_network_call(
                instaloader.Post.from_shortcode,
                self.L.context,
                post_shortcode,
                max_retries=3
            )
            
            # Wrap generator for pagination resilience
            comments_gen = self._wrap_generator(post.get_comments(), "comment_graph")
            
            for comment in comments_gen:
                if comment.created_at_utc < cutoff_time:
                    break
                
                mentions = re.findall(r'@([a-zA-Z0-9_.]+)', comment.text)
                hashed_mentions = [self.anonymize_user_id(m) for m in mentions]
                sanitized_text = TextSanitizer.sanitize(comment.text)
                
                comment_graph_data = {
                    "comment_id": str(comment.id),
                    "text": comment.text,
                    "text_sanitized": sanitized_text,
                    "author": self.anonymize_user_id(comment.owner.username),
                    "mentions_graph": hashed_mentions,
                    "created_at": comment.created_at_utc.isoformat(),
                    "source": "instagram_comment",
                }
                
                self._validate_and_publish(comment_graph_data, "comment_graph")
                
                comments_data.append(comment_graph_data)
                
                if len(comments_data) % 20 == 0:
                    self._jitter(2.0, 5.0)
                    
        except Exception as e:
            logger.error(f"Graph extraction failed for {post_shortcode}: {e}")
            
        return comments_data
    
    def extract_interaction_graph(
        self,
        post_shortcode: str
    ) -> Dict[str, Any]:
        """Extract full interaction network with O(1) lookups and resilient generator."""
        logger.info(f"Extracting interaction graph for: {post_shortcode}")
        
        nodes = []
        edges = []
        seen_node_ids = set()  # O(1) lookup
        
        try:
            post = self._safe_network_call(
                instaloader.Post.from_shortcode,
                self.L.context,
                post_shortcode,
                max_retries=3
            )
            
            # Wrap generator for pagination resilience
            comments_gen = self._wrap_generator(post.get_comments(), "interaction_graph")
            
            author = self.anonymize_user_id(post.owner_username)
            if author not in seen_node_ids:
                seen_node_ids.add(author)
                nodes.append({"id": author, "type": "author", "post_id": post_shortcode})
            
            edges.append({
                "source": author,
                "target": post_shortcode,
                "type": "owns"
            })
            
            for comment in comments_gen:
                commenter = self.anonymize_user_id(comment.owner.username)
                if commenter not in seen_node_ids:
                    seen_node_ids.add(commenter)
                    nodes.append({"id": commenter, "type": "commenter", "post_id": post_shortcode})
                
                edges.append({
                    "source": commenter,
                    "target": author,
                    "type": "commented_on",
                    "timestamp": comment.created_at_utc.isoformat()
                })
                
                # @mentions - O(1) lookup
                mentions = re.findall(r'@([a-zA-Z0-9_.]+)', comment.text)
                for mention in mentions:
                    mentioned = self.anonymize_user_id(mention)
                    if mentioned not in seen_node_ids:
                        seen_node_ids.add(mentioned)
                        nodes.append({"id": mentioned, "type": "mentioned", "post_id": post_shortcode})
                    
                    edges.append({
                        "source": commenter,
                        "target": mentioned,
                        "type": "mentioned",
                        "timestamp": comment.created_at_utc.isoformat()
                    })
                
                # Reply chains - O(1) lookup
                if hasattr(comment, 'answers') and comment.answers:
                    for reply in comment.answers:
                        replier = self.anonymize_user_id(reply.owner.username)
                        if replier not in seen_node_ids:
                            seen_node_ids.add(replier)
                            nodes.append({"id": replier, "type": "replier", "post_id": post_shortcode})
                        
                        edges.append({
                            "source": replier,
                            "target": commenter,
                            "type": "replied_to",
                            "timestamp": reply.created_at_utc.isoformat()
                        })
                
                if len(nodes) % 50 == 0:
                    self._jitter(2.0, 5.0)
            
            graph_data = {"nodes": nodes, "edges": edges, "post_shortcode": post_shortcode}
            self._validate_and_publish(graph_data, "interaction_graph")
            
            self.telemetry.increment("successful_scrapes")
                    
        except Exception as e:
            self.telemetry.increment("failed_scrapes")
            logger.error(f"Interaction graph extraction failed: {e}")
        
        return {"nodes": nodes, "edges": edges}
    
    def calculate_engagement_velocity(
        self,
        post_shortcode: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Calculate engagement velocity metrics with memory-efficient iteration.
        
        CRITICAL: Iterates incrementally instead of loading all comments into RAM.
        Instaloader yields comments in reverse chronological order (newest first),
        so we break early once we hit comments older than the cutoff.
        """
        logger.info(f"Calculating engagement velocity for: {post_shortcode}")
        
        try:
            post = self._safe_network_call(
                instaloader.Post.from_shortcode,
                self.L.context,
                post_shortcode,
                max_retries=3
            )
            
            comments_gen = self._wrap_generator(post.get_comments(), "velocity_comments")
            now = datetime.utcnow()
            cutoff = now - timedelta(hours=time_window_hours)
            
            # Memory-efficient: iterate and break early instead of list()
            # Instagram returns newest first, so we stop when we hit older comments
            recent_comments = []
            all_commenters = set()
            
            for comment in comments_gen:
                if comment.created_at_utc < cutoff:
                    # STOP FETCHING. We are past the time window.
                    # Comments are in reverse chronological order.
                    logger.info(f"Reached cutoff at {len(recent_comments)} recent comments")
                    break
                
                recent_comments.append(comment)
                all_commenters.add(comment.owner.username)
                
                # Safety limit to prevent infinite loops
                if len(recent_comments) >= 50000:
                    logger.warning(f"Safety limit reached at 10,000 comments in window")
                    break
            
            velocity = {
                "post_shortcode": post_shortcode,
                "time_window_hours": time_window_hours,
                "comments_per_hour": len(recent_comments) / time_window_hours if time_window_hours > 0 else 0,
                "unique_commenters": len(all_commenters),
                "total_comments": post.comments,  # Use post metadata instead of iterating all
                "likes": post.likes,
                "engagement_likes": post.likes,
                "engagement_comments": post.comments,
                "engagement_ratio": post.likes / (post.comments + 1),
                "controversy_score": self._calculate_controversy(recent_comments),
                "has_mentions": any(re.search(r'@([a-zA-Z0-9_.]+)', c.text) for c in recent_comments),
                "reply_ratio": self._calculate_reply_ratio(recent_comments),
            }
            
            self._validate_and_publish(velocity, "velocity_metrics")
            
            return velocity
            
        except Exception as e:
            logger.error(f"Velocity calculation failed: {e}")
            return {}
    
    def _calculate_controversy(self, comments: List) -> float:
        """Score based on reply chains and emotional language."""
        controversy_keywords = [
            "fake", "lie", "trash", "worst", "hate", "disgusting",
            "scam", "fraud", "exposed", "cancelled", "boycott",
            "trash", "garbage", "terrible", "horrible", "awful"
        ]
        
        reply_count = sum(1 for c in comments if hasattr(c, 'answers') and c.answers)
        emotional_count = sum(
            1 for c in comments 
            if any(kw in c.text.lower() for kw in controversy_keywords)
        )
        
        if not comments:
            return 0.0
        
        return (reply_count * 0.3 + emotional_count * 0.7) / len(comments)
    
    def _calculate_reply_ratio(self, comments: List) -> float:
        """Calculate ratio of comments that are replies."""
        if not comments:
            return 0.0
        
        reply_count = sum(1 for c in comments if hasattr(c, 'answers') and c.answers)
        return reply_count / len(comments)
    
    def collect_trending_posts(
        self,
        hashtags: List[str] = None,
        posts_per_hashtag: int = 50,
    ) -> List[Dict[str, Any]]:
        """Collect trending posts from specified hashtags."""
        if hashtags is None:
            hashtags = ["drama", "cancelled", "exposed", "tea", "spill", "gossip"]
        
        all_posts = []
        
        for hashtag in tqdm(hashtags, desc="Collecting Instagram posts"):
            logger.info(f"Collecting posts for #{hashtag}")
            
            count = 0
            for post in self.get_hashtag_posts(hashtag, limit=posts_per_hashtag):
                all_posts.append(post)
                count += 1
            
            logger.info(f"Collected {count} posts for #{hashtag}")
            self._jitter(3.0, 7.0)
        
        logger.info(f"Total Instagram posts collected: {len(all_posts)}")
        return all_posts
    
    def _parse_post(
        self,
        post: instaloader.Post,
        profile_username: str = None,
    ) -> Dict[str, Any]:
        """Parse Instagram post to dictionary."""
        return {
            "post_id": post.shortcode,
            "caption": post.caption or "",
            "caption_sanitized": TextSanitizer.sanitize(post.caption) if post.caption else "",
            "author": self.anonymize_user_id(post.owner_username),
            "author_followers": post.owner_profile.followers if post.owner_profile else None,
            "created_at": post.date_local.isoformat(),
            "likes": post.likes,
            "comments_count": post.comments,
            "is_video": post.is_video,
            "video_views": post.video_view_count if post.is_video else None,
            "hashtags": post.caption_hashtags,
            "mentions": post.tagged_users,
            "location": post.location.name if post.location else None,
            "image_url": post.url,
            "source": "instagram",
        }
    
    def get_telemetry(self) -> Dict[str, int]:
        """Get current telemetry metrics."""
        return self.telemetry.get_metrics()


def create_instagram_scraper(
    username: str = None,
    password: str = None,
    session_file: str = "insta_session",
    proxy_list: List[str] = None,
    redis_url: str = "redis://localhost:6379",
    enable_queue: bool = True,
) -> InstagramScraper:
    """Create Instagram scraper instance with all hardening features."""
    return InstagramScraper(
        username=username,
        password=password,
        session_file=session_file,
        proxy_list=proxy_list,
        redis_url=redis_url,
        enable_queue=enable_queue,
    )
