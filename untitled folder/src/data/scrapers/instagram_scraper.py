"""Instagram data scraper module using Instaloader."""

import hashlib
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from loguru import logger
from tqdm import tqdm

from src.config import get_env_var

# Import instaloader with error handling
try:
    import instaloader
    INSTAGRAM_AVAILABLE = True
except ImportError:
    INSTAGRAM_AVAILABLE = False
    logger.warning("instaloader not installed. Instagram scraping will not be available.")


class InstagramScraper:
    """Instagram scraper using Instaloader for public profiles."""
    
    def __init__(
        self,
        username: str = None,
        password: str = None,
    ):
        """Initialize Instagram scraper.
        
        Args:
            username: Instagram username (optional, for login)
            password: Instagram password (optional, for login)
        """
        if not INSTAGRAM_AVAILABLE:
            raise ImportError("instaloader is required for Instagram scraping. Install with: pip install instaloader")
        
        self.username = username or get_env_var("INSTAGRAM_USERNAME")
        self.password = password or get_env_var("INSTAGRAM_PASSWORD")
        
        # Initialize Instaloader
        self.L = instaloader.Instaloader(
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=True,
            save_metadata=False,
            compress_json=False,
        )
        
        # Login if credentials provided
        if self.username and self.password:
            try:
                self.L.login(self.username, self.password)
                logger.info("Instagram login successful")
            except Exception as e:
                logger.warning(f"Instagram login failed: {e}")
        else:
            logger.info("Instagram scraper initialized without login (public access only)")
    
    def anonymize_user_id(self, username: str) -> str:
        """Hash username for anonymization."""
        return hashlib.sha256(username.encode()).hexdigest()[:16]
    
    def get_profile_posts(
        self,
        target_username: str,
        limit: int = 100,
    ) -> Generator[Dict[str, Any], None, None]:
        """Get posts from a public Instagram profile.
        
        Args:
            target_username: Target Instagram username
            limit: Maximum posts to retrieve
            
        Yields:
            Post data dictionaries
        """
        logger.info(f"Fetching posts from Instagram profile: {target_username}")
        
        try:
            profile = instaloader.Profile.from_username(self.L.context, target_username)
            
            count = 0
            for post in profile.get_posts():
                if count >= limit:
                    break
                    
                yield self._parse_post(post, target_username)
                count += 1
                
        except Exception as e:
            logger.error(f"Error fetching Instagram profile: {e}")
    
    def get_hashtag_posts(
        self,
        hashtag: str,
        limit: int = 100,
    ) -> Generator[Dict[str, Any], None, None]:
        """Get posts with a specific hashtag.
        
        Args:
            hashtag: Hashtag to search (without #)
            limit: Maximum posts to retrieve
            
        Yields:
            Post data dictionaries
        """
        logger.info(f"Fetching Instagram posts with hashtag: #{hashtag}")
        
        try:
            hashtag_obj = instaloader.Hashtag.from_name(self.L.context, hashtag)
            
            count = 0
            for post in hashtag_obj.get_posts():
                if count >= limit:
                    break
                    
                yield self._parse_post(post)
                count += 1
                
        except Exception as e:
            logger.error(f"Error fetching hashtag posts: {e}")
    
    def get_post_comments(
        self,
        post_shortcode: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get comments from an Instagram post.
        
        Args:
            post_shortcode: Instagram post shortcode
            limit: Maximum comments to retrieve
            
        Returns:
            List of comment data dictionaries
        """
        logger.info(f"Fetching comments for Instagram post: {post_shortcode}")
        
        comments = []
        try:
            post = instaloader.Post.from_shortcode(self.L.context, post_shortcode)
            
            count = 0
            for comment in post.get_comments():
                if count >= limit:
                    break
                comments.append({
                    "comment_id": comment.id,
                    "text": comment.text,
                    "author": self.anonymize_user_id(comment.owner.username),
                    "created_at": datetime.fromtimestamp(comment.created_at_utc.timestamp()).isoformat(),
                    "source": "instagram_comment",
                })
                count += 1
                
        except Exception as e:
            logger.error(f"Error fetching Instagram comments: {e}")
            
        return comments
    
    def collect_trending_posts(
        self,
        hashtags: List[str] = None,
        posts_per_hashtag: int = 50,
    ) -> List[Dict[str, Any]]:
        """Collect trending posts from specified hashtags.
        
        Args:
            hashtags: List of hashtags to search
            posts_per_hashtag: Posts per hashtag
            
        Returns:
            List of collected posts
        """
        if hashtags is None:
            hashtags = ["drama", "cancelled", "exposed", "tea", "spill"]
        
        all_posts = []
        
        for hashtag in tqdm(hashtags, desc="Collecting Instagram posts"):
            logger.info(f"Collecting posts for #{hashtag}")
            
            count = 0
            for post in self.get_hashtag_posts(hashtag, limit=posts_per_hashtag):
                all_posts.append(post)
                count += 1
                
            logger.info(f"Collected {count} posts for #{hashtag}")
            
        logger.info(f"Total Instagram posts collected: {len(all_posts)}")
        return all_posts
    
    def _parse_post(
        self,
        post: instaloader.Post,
        profile_username: str = None,
    ) -> Dict[str, Any]:
        """Parse Instagram post to dictionary.
        
        Args:
            post: Instaloader Post object
            profile_username: Original profile username
            
        Returns:
            Parsed post dictionary
        """
        return {
            "post_id": post.shortcode,
            "caption": post.caption or "",
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


def create_instagram_scraper() -> InstagramScraper:
    """Create Instagram scraper instance."""
    return InstagramScraper()
