"""Twitter/X data scraper module.

This module provides a unified scraping interface for Twitter/X.

It supports two backends:
- Official Twitter API via tweepy (requires API keys / bearer token)
- Twikit (unofficial, cookie-based, no API keys)

The public surface of `TwitterScraper` is kept compatible with the original
implementation used in the doom-index project, but internally it can choose
whichever backend is available and better suited.

Usage notes
-----------
- For tweepy backend, set the usual env vars (TWITTER_API_KEY, TWITTER_API_SECRET,
  TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET, TWITTER_BEARER_TOKEN).
- For Twikit backend, you must provide login details (email/username + password
  + optionally 2FA and guest token) or load a previously exported session.

The scraper prefers Twikit when a valid Twikit session is available because
Twitter's official APIs are heavily restricted / paid.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, List, Optional

from loguru import logger
from tqdm import tqdm

from src.config import get_env_var

# ------------------------------
# Optional dependencies
# ------------------------------

# tweepy (official API)
try:  # pragma: no cover - optional dependency
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    tweepy = None  # type: ignore
    TWEEPY_AVAILABLE = False
    logger.info("tweepy not installed. Official Twitter API backend will be disabled.")

# twikit (unofficial web client)
try:  # pragma: no cover - optional dependency
    from twikit import Client as TwikitClient, Tweet as TwikitTweet, User as TwikitUser

    TWIKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TwikitClient = None  # type: ignore
    TwikitTweet = None  # type: ignore
    TwikitUser = None  # type: ignore
    TWIKIT_AVAILABLE = False
    logger.info("twikit not installed. Twikit backend will be disabled.")


# ------------------------------
# Backend configuration
# ------------------------------

@dataclass
class TwikitConfig:
    """Configuration for the Twikit backend.

    You can either:
    - Provide login credentials and let the scraper log in and persist a session, or
    - Provide a pre-saved session file path.
    - Provide a cookies file (exported from browser).
    """

    email: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    two_factor_secret: Optional[str] = None
    session_file: Optional[str] = None
    cookies_file: Optional[str] = None
    # Language can affect search results
    language: str = "en-US"
    # Search product: "Top" or "Latest" (Twitter's search products)
    search_product: str = "Top"


# ------------------------------
# Main scraper
# ------------------------------

class TwitterScraper:
    """Twitter/X scraper that can use either tweepy or Twikit as a backend.

    Public methods:
    - search_cancellation_events
    - get_user_timeline
    - get_tweet_replies
    - collect_cancellation_samples
    """

    # Nominal official API rate limits (requests per 15-min window)
    RATE_LIMITS = {
        "search_tweets": 180,  # Standard v1.1
        "user_timeline": 900,
        "user_info": 900,
    }

    def __init__(
        self,
        # Tweepy credentials (optional if you use only Twikit)
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_secret: Optional[str] = None,
        bearer_token: Optional[str] = None,
        # Twikit configuration (optional but recommended in 2025+)
        twikit_config: Optional[TwikitConfig] = None,
    ) -> None:
        # ---------------------
        # Resolve config
        # ---------------------
        self.api_key = api_key or get_env_var("TWITTER_API_KEY", required=False)
        self.api_secret = api_secret or get_env_var("TWITTER_API_SECRET", required=False)
        self.access_token = access_token or get_env_var("TWITTER_ACCESS_TOKEN", required=False)
        self.access_secret = access_secret or get_env_var("TWITTER_ACCESS_SECRET", required=False)
        self.bearer_token = bearer_token or get_env_var("TWITTER_BEARER_TOKEN", required=False)

        self.twikit_config = twikit_config or TwikitConfig(
            email=get_env_var("TWIKIT_EMAIL", required=False),
            username=get_env_var("TWIKIT_USERNAME", required=False),
            password=get_env_var("TWIKIT_PASSWORD", required=False),
            two_factor_secret=get_env_var("TWIKIT_2FA_SECRET", required=False),
            session_file=get_env_var("TWIKIT_SESSION_FILE", required=False),
            cookies_file=get_env_var("TWIKIT_COOKIES_FILE", required=False),
            search_product=get_env_var("TWIKIT_SEARCH_PRODUCT", required=False) or "Top",
        )

        # ---------------------
        # Initialize backends
        # ---------------------
        self._init_tweepy_backend()
        self._init_twikit_backend()

        if not self.client and not self.twikit_client:
            logger.warning(
                "No Twitter backend initialized. Install tweepy or twikit and configure credentials/cookies."
            )
        else:
            backends = []
            if self.client:
                backends.append("tweepy")
            if self.twikit_client:
                backends.append("twikit")
            logger.info(f"TwitterScraper initialized with backends: {', '.join(backends)}")

    # ------------------------------
    # Backend initialization helpers
    # ------------------------------

    def _init_tweepy_backend(self) -> None:
        """Initialize tweepy-based API clients if credentials and dependency exist."""
        self.api = None
        self.client = None

        if not TWEEPY_AVAILABLE:
            return

        if not (self.api_key and self.api_secret):
            # Not critical if using Twikit.
            logger.debug("Twitter API keys not configured; tweepy backend will be disabled.")
            return

        if self.access_token and self.access_secret:
            try:
                auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
                auth.set_access_token(self.access_token, self.access_secret)
                self.api = tweepy.API(auth, wait_on_rate_limit=True)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.error(f"Failed to initialize tweepy v1.1 API: {exc}")

        if self.bearer_token:
            try:
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    consumer_key=self.api_key,
                    consumer_secret=self.api_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_secret,
                    wait_on_rate_limit=True,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                logger.error(f"Failed to initialize tweepy v2 Client: {exc}")

    def _init_twikit_backend(self) -> None:
        """Initialize Twikit client if dependency and configuration are available.

        Twikit is cookie/session based. You can:
        - Load an existing session from file; or
        - Load cookies from browser export; or
        - Log in interactively once and then save the session.
        """
        self.twikit_client: Optional[TwikitClient] = None

        if not TWIKIT_AVAILABLE:
            return

        cfg = self.twikit_config
        if not cfg:
            return

        try:
            # New Twikit API doesn't have location parameter
            client = TwikitClient(language=cfg.language)

            # Try to load cookies file first (browser export)
            if cfg.cookies_file:
                try:
                    import json
                    with open(cfg.cookies_file, 'r') as f:
                        browser_cookies = json.load(f)
                    # Convert to Twikit format (dict with name as key)
                    cookies = {c['name']: c['value'] for c in browser_cookies}
                    client.set_cookies(cookies)
                    logger.info(f"Loaded Twikit cookies from {cfg.cookies_file}")
                    self.twikit_client = client
                    return
                except Exception as exc:
                    logger.warning(f"Failed to load cookies file: {exc}. Trying other methods.")

            if cfg.session_file:
                # Try to load pre-existing session first
                try:
                    client.load_session(cfg.session_file)
                    logger.info(f"Loaded Twikit session from {cfg.session_file}")
                    self.twikit_client = client
                    return
                except Exception as exc:  # pragma: no cover - filesystem/network
                    logger.warning(f"Failed to load Twikit session file: {exc}. Will try login if credentials exist.")

            # If we have credentials, attempt login
            if cfg.email and cfg.username and cfg.password:
                logger.info("Logging in to Twitter via Twikit.")
                # Twikit login is async; use asyncio.run for simplicity inside sync code.
                asyncio.run(self._twikit_login_and_maybe_persist(client, cfg))
                self.twikit_client = client
            else:
                logger.debug("Twikit credentials or session file not provided; Twikit backend disabled.")
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error(f"Failed to initialize Twikit backend: {exc}")

    async def _twikit_login_and_maybe_persist(self, client: TwikitClient, cfg: TwikitConfig) -> None:
        """Async helper to log in with Twikit and save session if a path is given."""
        try:
            # New Twikit API uses auth_info_1 and auth_info_2
            await client.login(
                auth_info_1=cfg.email or cfg.username,
                auth_info_2=cfg.username if cfg.email else None,
                password=cfg.password,
                totp_secret=cfg.two_factor_secret,
            )
            logger.info("Successfully logged in with Twikit.")
            if cfg.session_file:
                client.save_session(cfg.session_file)
                logger.info(f"Saved Twikit session to {cfg.session_file}")
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error(f"Twikit login failed: {exc}")

    # ------------------------------
    # Utility helpers
    # ------------------------------

    @staticmethod
    def anonymize_user_id(user_id: str) -> str:
        """Hash user ID for anonymization.

        The anonymized ID is a 16-character hex digest, stable for the same input.
        """

        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    # ------------------------------
    # High-level public API
    # ------------------------------

    def search_cancellation_events(
        self,
        query: str = "#cancel",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 100,
    ) -> Generator[Dict[str, Any], None, None]:
        """Search for cancellation-related tweets.

        Uses Twikit backend when available; falls back to tweepy search_all_tweets
        if Twikit cannot be used.

        Args:
            query: Search query (Twitter standard search syntax; extra filters added automatically).
            start_date: Start datetime for search (if backend supports it).
            end_date: End datetime for search (if backend supports it).
            max_results: Upper bound of results to attempt to retrieve.

        Yields:
            Normalized tweet dictionaries.
        """

        if self.twikit_client:
            logger.info(f"Searching tweets via Twikit with query: {query}")
            yield from self._twikit_search_cancellation_events(query, start_date, end_date, max_results)
            return

        if not (self.client and TWEEPY_AVAILABLE):
            logger.error("No Twitter backend available for search. Install twikit or configure tweepy credentials.")
            return

        logger.info(f"Searching tweets via tweepy with query: {query}")
        yield from self._tweepy_search_cancellation_events(query, start_date, end_date, max_results)

    def get_user_timeline(
        self,
        user_id: str,
        max_tweets: int = 200,
    ) -> List[Dict[str, Any]]:
        """Get user's tweet timeline.

        Tries Twikit first (can use username or user ID), then tweepy v2.
        """

        logger.info(f"Fetching timeline for user: {user_id}")

        # Twikit path
        if self.twikit_client:
            try:
                return list(self._twikit_get_user_timeline(user_id, max_tweets))
            except Exception as exc:  # pragma: no cover - network dependent
                logger.error(f"Error fetching user timeline via Twikit: {exc}")

        # Tweepy path
        if not (self.client and TWEEPY_AVAILABLE):
            logger.error("No Twitter backend available for user timeline.")
            return []

        return self._tweepy_get_user_timeline(user_id, max_tweets)

    def get_tweet_replies(
        self,
        tweet_id: str,
        max_replies: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get replies to a specific tweet.

        Twikit backend can use conversation lookups; we fall back to the existing
        tweepy implementation when needed.
        """

        logger.info(f"Fetching replies for tweet: {tweet_id}")

        if self.twikit_client:
            try:
                return list(self._twikit_get_tweet_replies(tweet_id, max_replies))
            except Exception as exc:  # pragma: no cover - network dependent
                logger.error(f"Error fetching replies via Twikit: {exc}")

        if not (self.client and TWEEPY_AVAILABLE):
            logger.error("No Twitter backend available for tweet replies.")
            return []

        return self._tweepy_get_tweet_replies(tweet_id, max_replies)

    def collect_cancellation_samples(
        self,
        keywords: Optional[List[str]] = None,
        samples_per_keyword: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Collect samples of cancellation-related events across a set of queries."""

        if keywords is None:
            keywords = [
                "#cancel",
                "#cancelled",
                "#cancellation",
                "cancelled after",
                "facing backlash",
                "called out for",
                "under fire for",
                "controversy",
            ]

        all_tweets: List[Dict[str, Any]] = []

        for keyword in tqdm(keywords, desc="Collecting tweets"):
            logger.info(f"Collecting tweets for: {keyword}")

            count = 0
            for tweet in self.search_cancellation_events(query=keyword, max_results=samples_per_keyword):
                all_tweets.append(tweet)
                count += 1
                if count >= samples_per_keyword:
                    break

            logger.info(f"Collected {count} tweets for {keyword}")

        logger.info(f"Total tweets collected: {len(all_tweets)}")
        return all_tweets

    def get_trends(
        self,
        location: str = "trending",
    ) -> List[Dict[str, Any]]:
        """Get trending topics from Twitter/X.

        Uses Twikit backend when available to fetch trending topics.

        Args:
            location: Location for trends (default: "trending").

        Returns:
            List of trending topic dictionaries with name, volume, and query info.
        """
        if self.twikit_client:
            logger.info(f"Fetching trends via Twikit for location: {location}")
            return self._twikit_get_trends(location)

        logger.warning("No Twikit backend available for trends. Install twikit and configure credentials.")
        return []

    # ------------------------------
    # Tweepy-based implementations
    # ------------------------------

    def _tweepy_search_cancellation_events(
        self,
        query: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        max_results: int,
    ) -> Iterable[Dict[str, Any]]:
        if not self.client or not TWEEPY_AVAILABLE:
            return []

        query_builder = f"{query} -is:retweet lang:en"

        fetched = 0
        try:
            for response in tweepy.Paginator(
                self.client.search_all_tweets,
                query=query_builder,
                start_time=start_date,
                end_time=end_date,
                max_results=min(max_results, 100),  # API limit 100
                tweet_fields=[
                    "created_at",
                    "public_metrics",
                    "author_id",
                    "context_annotations",
                    "entities",
                    "geo",
                ],
                expansions=["author_id", "referenced_tweets.id"],
                user_fields=["username", "public_metrics", "verified"],
            ):
                if not response.data:
                    continue

                for tweet in response.data:
                    yield self._parse_tweepy_tweet(tweet, response.includes)
                    fetched += 1
                    if fetched >= max_results:
                        return
        except tweepy.TweepyException as exc:  # pragma: no cover - network dependent
            logger.error(f"Twitter API error (tweepy): {exc}")
            raise

    def _tweepy_get_user_timeline(self, user_id: str, max_tweets: int) -> List[Dict[str, Any]]:
        if not self.client or not TWEEPY_AVAILABLE:
            return []

        tweets: List[Dict[str, Any]] = []

        try:
            for response in tweepy.Paginator(
                self.client.get_users_tweets,
                id=user_id,
                max_results=100,
                limit=max_tweets // 100 + 1,
                tweet_fields=["created_at", "public_metrics", "entities"],
            ):
                if not response.data:
                    continue

                for tweet in response.data:
                    tweets.append(self._parse_tweepy_tweet(tweet, includes=None))
                    if len(tweets) >= max_tweets:
                        return tweets
        except tweepy.TweepyException as exc:  # pragma: no cover - network dependent
            logger.error(f"Error fetching user timeline via tweepy: {exc}")

        return tweets

    def _tweepy_get_tweet_replies(self, tweet_id: str, max_replies: int) -> List[Dict[str, Any]]:
        if not self.client or not TWEEPY_AVAILABLE:
            return []

        replies: List[Dict[str, Any]] = []

        try:
            query = f"conversation_id:{tweet_id}"

            for response in tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=100,
                tweet_fields=["created_at", "public_metrics", "author_id", "entities"],
            ):
                if not response.data:
                    continue

                for tweet in response.data:
                    replies.append(self._parse_tweepy_tweet(tweet, includes=None))
                    if len(replies) >= max_replies:
                        return replies
        except tweepy.TweepyException as exc:  # pragma: no cover - network dependent
            logger.error(f"Error fetching replies via tweepy: {exc}")

        return replies

    def _parse_tweepy_tweet(self, tweet: Any, includes: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize a tweepy Tweet object into the project's tweet schema."""

        data: Dict[str, Any] = {
            "tweet_id": getattr(tweet, "id", None),
            "text": getattr(tweet, "text", None),
            "created_at": tweet.created_at.isoformat() if getattr(tweet, "created_at", None) else None,
            "author_id": self.anonymize_user_id(str(tweet.author_id)) if getattr(tweet, "author_id", None) else None,
            "metrics": {
                "likes": (tweet.public_metrics or {}).get("like_count", 0) if hasattr(tweet, "public_metrics") else 0,
                "retweets": (tweet.public_metrics or {}).get("retweet_count", 0) if hasattr(tweet, "public_metrics") else 0,
                "replies": (tweet.public_metrics or {}).get("reply_count", 0) if hasattr(tweet, "public_metrics") else 0,
                "quotes": (tweet.public_metrics or {}).get("quote_count", 0) if hasattr(tweet, "public_metrics") else 0,
            },
            "hashtags": [
                tag["tag"]
                for tag in (getattr(tweet, "entities", {}) or {}).get("hashtags", [])
            ],
            "mentions": [
                m["username"]
                for m in (getattr(tweet, "entities", {}) or {}).get("mentions", [])
            ],
            "source": "twitter",
        }

        # Add user info if available
        if includes and "users" in includes and getattr(tweet, "author_id", None):
            for user in includes["users"]:
                if getattr(user, "id", None) == tweet.author_id:
                    data["user"] = {
                        "username": getattr(user, "username", None),
                        "followers": (getattr(user, "public_metrics", {}) or {}).get("followers_count", 0),
                        "verified": bool(getattr(user, "verified", False)),
                    }
                    break

        return data

    # ------------------------------
    # Twikit-based implementations
    # ------------------------------

    def _twikit_search_cancellation_events(
        self,
        query: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        max_results: int,
    ) -> Iterable[Dict[str, Any]]:
        """Search using Twikit's full-archive search.

        Twikit does not expose an identical API to tweepy, but we can approximate
        via client.search_tweets and then filter dates in Python if needed.
        """

        client = self.twikit_client
        if client is None:
            return []

        # Twikit uses async methods; create a small async driver and run it.
        async def _runner() -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []

            # Use configurable search product (Top or Latest) from TwikitConfig
            search_product = getattr(self.twikit_config, 'search_product', 'Top')
            # Twikit's search_tweets returns a TweetList that supports pagination via .next()
            # Reference: https://twikit.readthedocs.io/en/latest/twikit.html#twikit.client.Client.search_tweets
            tweet_list = await client.search_tweets(query=query, product=search_product)

            async def _consume_list(tweets: Iterable[TwikitTweet]) -> None:
                nonlocal results
                for t in tweets:
                    if start_date or end_date:
                        created_at = self._twikit_parse_datetime(t)
                        if start_date and created_at and created_at < start_date:
                            continue
                        if end_date and created_at and created_at > end_date:
                            continue
                    results.append(self._parse_twikit_tweet(t))
                    if len(results) >= max_results:
                        return

            # First page
            await _consume_list(tweet_list)
            if len(results) >= max_results:
                return results

            # Paginate while Twikit has more
            while tweet_list.has_next_page and len(results) < max_results:
                tweet_list = await tweet_list.next()
                await _consume_list(tweet_list)

            return results

        try:
            tweets = asyncio.run(_runner())
        except RuntimeError:
            # If an event loop is already running (e.g. inside async context), nest via new loop
            loop = asyncio.new_event_loop()
            try:
                tweets = loop.run_until_complete(_runner())
            finally:
                loop.close()
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error(f"Error searching tweets via Twikit: {exc}")
            tweets = []

        for t in tweets:
            yield t

    def _twikit_get_user_timeline(self, user_id: str, max_tweets: int) -> Iterable[Dict[str, Any]]:
        client = self.twikit_client
        if client is None:
            return []

        async def _runner() -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []

            # user_id may actually be a username; Twikit can resolve both via get_user_by_screen_name
            try:
                if user_id.isdigit():
                    user = await client.get_user_by_id(user_id)
                else:
                    user = await client.get_user_by_screen_name(user_id)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.error(f"Twikit: failed to resolve user '{user_id}': {exc}")
                return results

            tweet_list = await user.get_tweets("Tweets")

            async def _consume_list(tweets: Iterable[TwikitTweet]) -> None:
                nonlocal results
                for t in tweets:
                    results.append(self._parse_twikit_tweet(t))
                    if len(results) >= max_tweets:
                        return

            await _consume_list(tweet_list)
            if len(results) >= max_tweets:
                return results

            while tweet_list.has_next_page and len(results) < max_tweets:
                tweet_list = await tweet_list.next()
                await _consume_list(tweet_list)

            return results

        try:
            tweets = asyncio.run(_runner())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                tweets = loop.run_until_complete(_runner())
            finally:
                loop.close()
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error(f"Error fetching user timeline via Twikit: {exc}")
            tweets = []

        return tweets

    def _twikit_get_tweet_replies(self, tweet_id: str, max_replies: int) -> Iterable[Dict[str, Any]]:
        client = self.twikit_client
        if client is None:
            return []

        async def _runner() -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []

            try:
                tweet: TwikitTweet = await client.get_tweet_by_id(tweet_id)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.error(f"Twikit: failed to get tweet {tweet_id}: {exc}")
                return results

            reply_list = await tweet.get_replies()

            async def _consume_list(tweets: Iterable[TwikitTweet]) -> None:
                nonlocal results
                for t in tweets:
                    results.append(self._parse_twikit_tweet(t))
                    if len(results) >= max_replies:
                        return

            await _consume_list(reply_list)
            if len(results) >= max_replies:
                return results

            while reply_list.has_next_page and len(results) < max_replies:
                reply_list = await reply_list.next()
                await _consume_list(reply_list)

            return results

        try:
            replies = asyncio.run(_runner())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                replies = loop.run_until_complete(_runner())
            finally:
                loop.close()
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error(f"Error fetching tweet replies via Twikit: {exc}")
            replies = []

        return replies

    def _twikit_get_trends(self, location: str = "trending") -> List[Dict[str, Any]]:
        """Get trending topics using Twikit backend.
        
        New Twikit API requires category parameter.
        """
        client = self.twikit_client
        if client is None:
            return []

        async def _runner() -> List[Dict[str, Any]]:
            try:
                # Map location to category
                category_map = {
                    "trending": "trending",
                    "for-you": "for-you", 
                    "news": "news",
                    "sports": "sports",
                    "entertainment": "entertainment"
                }
                category = category_map.get(location, "trending")
                trends = await client.get_trends(category=category)
                results: List[Dict[str, Any]] = []
                for trend in trends:
                    results.append({
                        "name": getattr(trend, "name", None),
                        "domain": getattr(trend, "domain_context", None),
                        "url": getattr(trend, "url", None),
                        "tweet_volume": getattr(trend, "tweets_count", None),
                        "location": location,
                    })
                return results
            except Exception as exc:  # pragma: no cover - network dependent
                logger.error(f"Error fetching trends via Twikit: {exc}")
                return []

        try:
            trends = asyncio.run(_runner())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                trends = loop.run_until_complete(_runner())
            finally:
                loop.close()
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error(f"Error fetching trends via Twikit: {exc}")
            trends = []

        return trends

    @staticmethod
    def _twikit_parse_datetime(tweet: TwikitTweet) -> Optional[datetime]:  # type: ignore[valid-type]
        """Parse Twikit tweet datetime.

        Twikit Tweet objects expose .created_at as `datetime` already in recent
        versions; we gracefully handle string cases as well.
        """

        created = getattr(tweet, "created_at", None)
        if isinstance(created, datetime):
            return created
        if isinstance(created, str):  # pragma: no cover - depends on version
            try:
                # Twikit typically uses ISO 8601 format
                return datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    def _parse_twikit_tweet(self, tweet: TwikitTweet) -> Dict[str, Any]:  # type: ignore[valid-type]
        """Normalize a Twikit Tweet object into the project's tweet schema."""

        created_at = self._twikit_parse_datetime(tweet)

        # Public metrics may not be grouped; Twikit exposes properties directly
        # such as favorite_count, retweet_count, reply_count, quote_count.
        data: Dict[str, Any] = {
            "tweet_id": getattr(tweet, "id", None) or getattr(tweet, "tweet_id", None),
            "text": getattr(tweet, "text", None) or getattr(tweet, "full_text", None),
            "created_at": created_at.isoformat() if created_at else None,
            "author_id": self.anonymize_user_id(str(getattr(tweet, "user_id", "")))
            if getattr(tweet, "user_id", None)
            else None,
            "metrics": {
                "likes": getattr(tweet, "favorite_count", 0),
                "retweets": getattr(tweet, "retweet_count", 0),
                "replies": getattr(tweet, "reply_count", 0),
                "quotes": getattr(tweet, "quote_count", 0),
            },
            # Twikit entity parsing can vary; we try best-effort extraction.
            "hashtags": self._extract_twikit_hashtags(tweet),
            "mentions": self._extract_twikit_mentions(tweet),
            "source": "twitter",
        }

        # Attach user info if available on the tweet object
        user_obj = getattr(tweet, "user", None)
        if user_obj is not None:
            data["user"] = {
                "username": getattr(user_obj, "screen_name", None) or getattr(user_obj, "username", None),
                "followers": getattr(user_obj, "followers_count", 0),
                "verified": bool(getattr(user_obj, "verified", False)),
            }

        return data

    @staticmethod
    def _extract_twikit_hashtags(tweet: TwikitTweet) -> List[str]:  # type: ignore[valid-type]
        entities = getattr(tweet, "entities", None) or {}
        hashtags_raw = entities.get("hashtags", []) if isinstance(entities, dict) else []
        tags: List[str] = []
        for h in hashtags_raw:
            if isinstance(h, dict) and "text" in h:
                tags.append(h["text"])
            elif hasattr(h, "text"):
                tags.append(getattr(h, "text"))
        return tags

    @staticmethod
    def _extract_twikit_mentions(tweet: TwikitTweet) -> List[str]:  # type: ignore[valid-type]
        entities = getattr(tweet, "entities", None) or {}
        mentions_raw = entities.get("user_mentions", []) if isinstance(entities, dict) else []
        users: List[str] = []
        for m in mentions_raw:
            if isinstance(m, dict) and "screen_name" in m:
                users.append(m["screen_name"])
            elif hasattr(m, "screen_name"):
                users.append(getattr(m, "screen_name"))
        return users

    # ------------------------------
    # New Twikit methods for Timeline and Search
    # ------------------------------

    def get_home_timeline(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get home timeline tweets using Twikit backend.
        
        Args:
            count: Number of tweets to fetch (default: 100)
            
        Returns:
            List of tweet dictionaries.
        """
        if not self.twikit_client:
            logger.warning("No Twikit backend available for timeline.")
            return []
        
        return self._twikit_get_home_timeline(count)

    def _twikit_get_home_timeline(self, count: int) -> List[Dict[str, Any]]:
        """Get home timeline using Twikit backend."""
        client = self.twikit_client
        if client is None:
            return []

        async def _runner() -> List[Dict[str, Any]]:
            try:
                timeline = await client.get_latest_timeline(count=count)
                results: List[Dict[str, Any]] = []
                for tweet in timeline:
                    results.append({
                        "id": tweet.id,
                        "user": tweet.user.screen_name,
                        "user_name": tweet.user.name,
                        "text": tweet.text,
                        "created_at": str(tweet.created_at) if tweet.created_at else None,
                        "likes": tweet.favorite_count,
                        "retweets": tweet.retweet_count,
                        "replies": tweet.reply_count,
                    })
                return results
            except Exception as exc:
                logger.error(f"Error fetching timeline via Twikit: {exc}")
                return []

        try:
            return asyncio.run(_runner())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_runner())

    def search_tweets(self, query: str, product: str = "Top", count: int = 20) -> List[Dict[str, Any]]:
        """Search for tweets using Twikit backend.
        
        Args:
            query: Search query
            product: Search product ("Top", "Latest", or "Media")
            count: Number of results to fetch
            
        Returns:
            List of tweet dictionaries.
        """
        if not self.twikit_client:
            logger.warning("No Twikit backend available for search.")
            return []
        
        return self._twikit_search_tweets(query, product, count)

    def _twikit_search_tweets(self, query: str, product: str, count: int) -> List[Dict[str, Any]]:
        """Search tweets using Twikit backend."""
        client = self.twikit_client
        if client is None:
            return []

        async def _runner() -> List[Dict[str, Any]]:
            try:
                results = await client.search_tweet(query, product=product, count=count)
                tweets: List[Dict[str, Any]] = []
                for tweet in results:
                    tweets.append({
                        "id": tweet.id,
                        "user": tweet.user.screen_name,
                        "user_name": tweet.user.name,
                        "text": tweet.text,
                        "created_at": str(tweet.created_at) if tweet.created_at else None,
                        "likes": tweet.favorite_count,
                        "retweets": tweet.retweet_count,
                    })
                return tweets
            except Exception as exc:
                logger.error(f"Error searching tweets via Twikit: {exc}")
                return []

        try:
            return asyncio.run(_runner())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_runner())


# ------------------------------
# Factory
# ------------------------------


def create_twitter_scraper() -> TwitterScraper:
    """Create a TwitterScraper instance.

    This is kept for backwards compatibility with the rest of the project.
    """

    return TwitterScraper()
