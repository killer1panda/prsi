"""Twitter/X data scraper module."""

import hashlib
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from loguru import logger
from tqdm import tqdm

from src.config import get_env_var

# Import tweepy with error handling
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    logger.warning("tweepy not installed. Twitter scraping will not be available.")


class TwitterScraper:
    """Twitter API scraper with rate limiting and error handling."""
    
    # Rate limits (requests per 15-min window)
    RATE_LIMITS = {
        "search_tweets": 180,  # Standard
        "user_timeline": 900,
        "user_info": 900,
    }
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        access_token: str = None,
        access_secret: str = None,
        bearer_token: str = None,
    ):
        """Initialize Twitter API client.
        
        Args:
            api_key: Twitter API key
            api_secret: Twitter API secret
            access_token: Twitter access token
            access_secret: Twitter access token secret
            bearer_token: Twitter bearer token (for v2 API)
        """
        if not TWITTER_AVAILABLE:
            raise ImportError("tweepy is required for Twitter scraping. Install with: pip install tweepy")
        
        self.api_key = api_key or get_env_var("TWITTER_API_KEY")
        self.api_secret = api_secret or get_env_var("TWITTER_API_SECRET")
        self.access_token = access_token or get_env_var("TWITTER_ACCESS_TOKEN")
        self.access_secret = access_secret or get_env_var("TWITTER_ACCESS_SECRET")
        self.bearer_token = bearer_token or get_env_var("TWITTER_BEARER_TOKEN")
        
        # Check if credentials are available
        if not all([self.api_key, self.api_secret, self.bearer_token]):
            logger.warning("Twitter API credentials not fully configured. Some features may not work.")
        
        # Initialize API v1.1 client
        if self.api_key and self.api_secret and self.access_token and self.access_secret:
            auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
            auth.set_access_token(self.access_token, self.access_secret)
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
        else:
            self.api = None
        
        # Initialize API v2 client
        if self.bearer_token:
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_secret,
                wait_on_rate_limit=True
            )
        else:
            self.client = None
        
        logger.info("Twitter API client initialized")
    
    def anonymize_user_id(self, user_id: str) -> str:
        """Hash user ID for anonymization.
        
        Args:
            user_id: Original user ID
            
        Returns:
            Hashed user ID
        """
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def search_cancellation_events(
        self,
        query: str = "#cancel",
        start_date: datetime = None,
        end_date: datetime = None,
        max_results: int = 100,
    ) -> Generator[Dict[str, Any], None, None]:
        """Search for cancellation-related tweets.
        
        Args:
            query: Search query
            start_date: Start date for search
            end_date: End date for search
            max_results: Maximum results per request (10-100)
            
        Yields:
            Tweet data dictionaries
        """
        if not self.client:
            logger.error("Twitter v2 API client not initialized. Check your bearer token.")
            return
        
        logger.info(f"Searching tweets with query: {query}")
        
        query_builder = f"{query} -is:retweet lang:en"
        
        try:
            for response in tweepy.Paginator(
                self.client.search_all_tweets,
                query=query_builder,
                start_time=start_date,
                end_time=end_date,
                max_results=max_results,
                tweet_fields=[
                    "created_at", "public_metrics", "author_id",
                    "context_annotations", "entities", "geo"
                ],
                expansions=["author_id", "referenced_tweets.id"],
                user_fields=["username", "public_metrics", "verified"],
            ):
                if response.data:
                    for tweet in response.data:
                        yield self._parse_tweet(tweet, response.includes)
                        
        except tweepy.TweepyException as e:
            logger.error(f"Twitter API error: {e}")
            raise
    
    def get_user_timeline(
        self,
        user_id: str,
        max_tweets: int = 200,
    ) -> List[Dict[str, Any]]:
        """Get user's tweet timeline.
        
        Args:
            user_id: Twitter user ID
            max_tweets: Maximum tweets to retrieve
            
        Returns:
            List of tweet data dictionaries
        """
        if not self.client:
            logger.error("Twitter v2 API client not initialized")
            return []
        
        logger.info(f"Fetching timeline for user: {user_id}")
        
        tweets = []
        try:
            for response in tweepy.Paginator(
                self.client.get_users_tweets,
                id=user_id,
                max_results=100,
                limit=max_tweets // 100 + 1,
                tweet_fields=["created_at", "public_metrics", "entities"],
            ):
                if response.data:
                    for tweet in response.data:
                        tweets.append(self._parse_tweet(tweet))
                        
        except tweepy.TweepyException as e:
            logger.error(f"Error fetching user timeline: {e}")
            
        return tweets
    
    def get_tweet_replies(
        self,
        tweet_id: str,
        max_replies: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get replies to a specific tweet.
        
        Args:
            tweet_id: Tweet ID
            max_replies: Maximum replies to retrieve
            
        Returns:
            List of reply data dictionaries
        """
        if not self.client:
            logger.error("Twitter v2 API client not initialized")
            return []
        
        logger.info(f"Fetching replies for tweet: {tweet_id}")
        
        replies = []
        try:
            # Search for replies using conversation_id
            query = f"conversation_id:{tweet_id}"
            
            for response in tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=100,
                tweet_fields=["created_at", "public_metrics", "author_id"],
            ):
                if response.data:
                    for tweet in response.data:
                        replies.append(self._parse_tweet(tweet))
                        
                if len(replies) >= max_replies:
                    break
                    
        except tweepy.TweepyException as e:
            logger.error(f"Error fetching replies: {e}")
            
        return replies
    
    def _parse_tweet(
        self,
        tweet,
        includes: Dict = None
    ) -> Dict[str, Any]:
        """Parse tweet object to dictionary.
        
        Args:
            tweet: Tweet object from API
            includes: Additional included data
            
        Returns:
            Parsed tweet dictionary
        """
        data = {
            "tweet_id": tweet.id,
            "text": tweet.text,
            "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
            "author_id": self.anonymize_user_id(str(tweet.author_id)) if tweet.author_id else None,
            "metrics": {
                "likes": tweet.public_metrics.get("like_count", 0) if tweet.public_metrics else 0,
                "retweets": tweet.public_metrics.get("retweet_count", 0) if tweet.public_metrics else 0,
                "replies": tweet.public_metrics.get("reply_count", 0) if tweet.public_metrics else 0,
                "quotes": tweet.public_metrics.get("quote_count", 0) if tweet.public_metrics else 0,
            },
            "hashtags": [tag["tag"] for tag in tweet.entities.get("hashtags", [])] if tweet.entities else [],
            "mentions": [m["username"] for m in tweet.entities.get("mentions", [])] if tweet.entities else [],
            "source": "twitter",
        }
        
        # Add user info if available
        if includes and "users" in includes:
            for user in includes["users"]:
                if user.id == tweet.author_id:
                    data["user"] = {
                        "username": user.username,
                        "followers": user.public_metrics.get("followers_count", 0) if user.public_metrics else 0,
                        "verified": user.verified or False,
                    }
                    break
                    
        return data
    
    def collect_cancellation_samples(
        self,
        keywords: List[str] = None,
        samples_per_keyword: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Collect samples of cancellation events.
        
        Args:
            keywords: List of cancellation-related keywords
            samples_per_keyword: Samples to collect per keyword
            
        Returns:
            List of collected tweets
        """
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
        
        all_tweets = []
        
        for keyword in tqdm(keywords, desc="Collecting tweets"):
            logger.info(f"Collecting tweets for: {keyword}")
            
            count = 0
            for tweet in self.search_cancellation_events(query=keyword):
                all_tweets.append(tweet)
                count += 1
                
                if count >= samples_per_keyword:
                    break
                    
            logger.info(f"Collected {count} tweets for {keyword}")
            
        logger.info(f"Total tweets collected: {len(all_tweets)}")
        return all_tweets


def create_twitter_scraper() -> TwitterScraper:
    """Create Twitter scraper instance."""
    return TwitterScraper()
