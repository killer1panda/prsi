"""Reddit data scraper module."""

import hashlib
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from loguru import logger
from tqdm import tqdm

from src.config import get_env_var

# Import PRAW with error handling
try:
    import praw
    from praw.models import Comment, Submission
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    logger.warning("praw not installed. Reddit scraping will not be available.")


class RedditScraper:
    """Reddit API scraper with rate limiting and error handling."""
    
    # Target subreddits for cancellation content
    CANCELLATION_SUBREDDITS = [
        "SubredditDrama",
        "OutOfTheLoop",
        "AmItheAsshole",
        "PublicFreakout",
        "entertainment",
        "news",
        "politics",
        "Fauxmoi",
        "HobbyDrama",
    ]
    
    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        user_agent: str = None,
    ):
        """Initialize Reddit API client.
        
        Args:
            client_id: Reddit client ID
            client_secret: Reddit client secret
            user_agent: Reddit user agent string
        """
        if not REDDIT_AVAILABLE:
            raise ImportError("praw is required for Reddit scraping. Install with: pip install praw")
        
        self.client_id = client_id or get_env_var("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or get_env_var("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or get_env_var("REDDIT_USER_AGENT", "doom-index/0.1.0")
        
        # Check if credentials are available
        if not all([self.client_id, self.client_secret]):
            logger.warning("Reddit API credentials not configured. Some features may not work.")
        
        # Initialize PRAW client
        if self.client_id and self.client_secret:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
        else:
            self.reddit = None
        
        logger.info("Reddit API client initialized")
    
    def anonymize_user_id(self, username: str) -> str:
        """Hash username for anonymization.
        
        Args:
            username: Original username
            
        Returns:
            Hashed username
        """
        if username in ["[deleted]", "AutoModerator"]:
            return username
        return hashlib.sha256(username.encode()).hexdigest()[:16]
    
    def get_subreddit_posts(
        self,
        subreddit_name: str,
        category: str = "hot",
        limit: int = 100,
        time_filter: str = "month",
    ) -> Generator[Dict[str, Any], None, None]:
        """Get posts from a subreddit.
        
        Args:
            subreddit_name: Name of subreddit
            category: Category (hot, new, top, controversial)
            limit: Maximum posts to retrieve
            time_filter: Time filter for top/controversial (hour, day, week, month, year, all)
            
        Yields:
            Post data dictionaries
        """
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return
        
        logger.info(f"Fetching {category} posts from r/{subreddit_name}")
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            if category == "hot":
                posts = subreddit.hot(limit=limit)
            elif category == "new":
                posts = subreddit.new(limit=limit)
            elif category == "top":
                posts = subreddit.top(time_filter=time_filter, limit=limit)
            elif category == "controversial":
                posts = subreddit.controversial(time_filter=time_filter, limit=limit)
            else:
                posts = subreddit.hot(limit=limit)
            
            for post in posts:
                yield self._parse_submission(post)
                
        except Exception as e:
            logger.error(f"Error fetching from r/{subreddit_name}: {e}")
    
    def get_post_comments(
        self,
        post_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get comments from a post.
        
        Args:
            post_id: Reddit post ID
            limit: Maximum comments to retrieve
            
        Returns:
            List of comment data dictionaries
        """
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return []
        
        logger.info(f"Fetching comments for post: {post_id}")
        
        comments = []
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Remove "load more" stubs
            
            for comment in submission.comments[:limit]:
                comments.append(self._parse_comment(comment))
                
        except Exception as e:
            logger.error(f"Error fetching comments: {e}")
            
        return comments
    
    def search_cancellation_posts(
        self,
        query: str = "cancelled",
        subreddit: str = "all",
        limit: int = 100,
        time_filter: str = "month",
    ) -> Generator[Dict[str, Any], None, None]:
        """Search for cancellation-related posts.
        
        Args:
            query: Search query
            subreddit: Subreddit to search (default: all)
            limit: Maximum results
            time_filter: Time filter
            
        Yields:
            Post data dictionaries
        """
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return
        
        logger.info(f"Searching Reddit for: {query}")
        
        try:
            search_results = self.reddit.subreddit(subreddit).search(
                query,
                sort="relevance",
                time_filter=time_filter,
                limit=limit,
            )
            
            for post in search_results:
                yield self._parse_submission(post)
                
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
    
    def get_user_history(
        self,
        username: str,
        limit: int = 100,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get user's post and comment history.
        
        Args:
            username: Reddit username
            limit: Maximum items per category
            
        Returns:
            Dictionary with posts and comments
        """
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return {"posts": [], "comments": []}
        
        logger.info(f"Fetching history for user: {username}")
        
        result = {"posts": [], "comments": []}
        
        try:
            redditor = self.reddit.redditor(username)
            
            # Get posts
            for post in redditor.submissions.new(limit=limit):
                result["posts"].append(self._parse_submission(post))
            
            # Get comments
            for comment in redditor.comments.new(limit=limit):
                result["comments"].append(self._parse_comment(comment))
                
        except Exception as e:
            logger.error(f"Error fetching user history: {e}")
            
        return result
    
    def collect_drama_threads(
        self,
        samples_per_subreddit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Collect drama/cancellation threads from relevant subreddits.
        
        Args:
            samples_per_subreddit: Samples per subreddit
            
        Returns:
            List of collected posts with comments
        """
        all_posts = []
        
        for subreddit_name in tqdm(self.CANCELLATION_SUBREDDITS, desc="Collecting Reddit posts"):
            logger.info(f"Collecting from r/{subreddit_name}")
            
            count = 0
            for post in self.get_subreddit_posts(
                subreddit_name,
                category="hot",
                limit=samples_per_subreddit,
            ):
                # Get top comments for each post
                if post["num_comments"] > 10:
                    post["comments"] = self.get_post_comments(
                        post["post_id"],
                        limit=50
                    )
                
                all_posts.append(post)
                count += 1
                
            logger.info(f"Collected {count} posts from r/{subreddit_name}")
            
        logger.info(f"Total Reddit posts collected: {len(all_posts)}")
        return all_posts
    
    def _parse_submission(self, submission: Submission) -> Dict[str, Any]:
        """Parse submission object to dictionary.
        
        Args:
            submission: PRAW Submission object
            
        Returns:
            Parsed submission dictionary
        """
        return {
            "post_id": submission.id,
            "title": submission.title,
            "text": submission.selftext,
            "author": self.anonymize_user_id(str(submission.author)) if submission.author else "[deleted]",
            "subreddit": submission.subreddit.display_name,
            "created_at": datetime.fromtimestamp(submission.created_utc).isoformat(),
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "awards": submission.total_awards_received,
            "url": f"https://reddit.com{submission.permalink}",
            "is_video": submission.is_video,
            "link_flair_text": submission.link_flair_text,
            "source": "reddit",
        }
    
    def _parse_comment(self, comment: Comment) -> Dict[str, Any]:
        """Parse comment object to dictionary.
        
        Args:
            comment: PRAW Comment object
            
        Returns:
            Parsed comment dictionary
        """
        return {
            "comment_id": comment.id,
            "post_id": comment.submission.id,
            "text": comment.body,
            "author": self.anonymize_user_id(str(comment.author)) if comment.author else "[deleted]",
            "created_at": datetime.fromtimestamp(comment.created_utc).isoformat(),
            "score": comment.score,
            "is_submitter": comment.is_submitter,
            "parent_id": comment.parent_id,
            "source": "reddit_comment",
        }


def create_reddit_scraper() -> RedditScraper:
    """Create Reddit scraper instance."""
    return RedditScraper()
