"""Unified data collection pipeline."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from tqdm import tqdm

from src.config import config
from src.data.db_connectors import get_mongodb
from src.data.neo4j_connector import get_neo4j
from src.data.preprocessing import preprocess_posts


class DataCollectionPipeline:
    """Unified data collection pipeline for all platforms."""
    
    def __init__(
        self,
        target_samples: int = 50000,
        batch_size: int = 100,
    ):
        """Initialize data collection pipeline.
        
        Args:
            target_samples: Target number of samples
            batch_size: Batch size for database inserts
        """
        self.target_samples = target_samples
        self.batch_size = batch_size
        
        # Initialize databases
        self.mongodb = None
        self.neo4j = None
        
        # Scrapers will be initialized lazily
        self._twitter = None
        self._reddit = None
        self._instagram = None
        
        logger.info(f"Data collection pipeline initialized (target: {target_samples} samples)")
    
    @property
    def twitter(self):
        """Lazy load Twitter scraper."""
        if self._twitter is None:
            try:
                from src.data.scrapers.twitter_scraper import create_twitter_scraper
                self._twitter = create_twitter_scraper()
            except Exception as e:
                logger.warning(f"Twitter scraper not available: {e}")
        return self._twitter
    
    @property
    def reddit(self):
        """Lazy load Reddit scraper."""
        if self._reddit is None:
            try:
                from src.data.scrapers.reddit_scraper import create_reddit_scraper
                self._reddit = create_reddit_scraper()
            except Exception as e:
                logger.warning(f"Reddit scraper not available: {e}")
        return self._reddit
    
    @property
    def instagram(self):
        """Lazy load Instagram scraper."""
        if self._instagram is None:
            try:
                from src.data.scrapers.instagram_scraper import create_instagram_scraper
                self._instagram = create_instagram_scraper()
            except Exception as e:
                logger.warning(f"Instagram scraper not available: {e}")
        return self._instagram
    
    def _init_databases(self):
        """Initialize database connections."""
        if self.mongodb is None:
            self.mongodb = get_mongodb()
        if self.neo4j is None:
            self.neo4j = get_neo4j()
    
    def run_full_collection(self):
        """Run full data collection from all sources."""
        logger.info("Starting full data collection")
        
        self._init_databases()
        
        total_collected = 0
        
        # Collect from Twitter
        if self.twitter:
            try:
                twitter_samples = self._collect_twitter()
                total_collected += len(twitter_samples)
            except Exception as e:
                logger.error(f"Twitter collection failed: {e}")
        
        # Collect from Reddit
        if self.reddit:
            try:
                reddit_samples = self._collect_reddit()
                total_collected += len(reddit_samples)
            except Exception as e:
                logger.error(f"Reddit collection failed: {e}")
        
        # Collect from Instagram (limited)
        if self.instagram:
            try:
                instagram_samples = self._collect_instagram()
                total_collected += len(instagram_samples)
            except Exception as e:
                logger.error(f"Instagram collection failed: {e}")
        
        logger.info(f"Total samples collected: {total_collected}")
        
        # Generate collection report
        self._generate_report(total_collected)
    
    def _collect_twitter(self) -> List[Dict[str, Any]]:
        """Collect data from Twitter."""
        logger.info("Starting Twitter collection")
        
        tweets = self.twitter.collect_cancellation_samples(
            samples_per_keyword=500,
        )
        
        tweets = list(tweets)
        
        # Store in MongoDB
        if tweets:
            inserted = self.mongodb.insert_posts_batch(tweets)
            logger.info(f"Inserted {inserted} tweets into MongoDB")
            
            # Create graph relationships in Neo4j
            self._create_graph_relationships(tweets, "twitter")
        
        return tweets
    
    def _collect_reddit(self) -> List[Dict[str, Any]]:
        """Collect data from Reddit."""
        logger.info("Starting Reddit collection")
        
        posts = self.reddit.collect_drama_threads(
            samples_per_subreddit=100,
        )
        
        # Store in MongoDB
        if posts:
            inserted = self.mongodb.insert_posts_batch(posts)
            logger.info(f"Inserted {inserted} Reddit posts into MongoDB")
            
            # Create graph relationships in Neo4j
            self._create_graph_relationships(posts, "reddit")
        
        return posts
    
    def _collect_instagram(self) -> List[Dict[str, Any]]:
        """Collect data from Instagram (limited due to API restrictions)."""
        logger.info("Starting Instagram collection (limited)")
        
        # Collect from specific hashtags
        posts = self.instagram.collect_trending_posts(
            hashtags=["drama", "cancelled", "exposed"],
            posts_per_hashtag=50,
        )
        
        # Store in MongoDB
        if posts:
            inserted = self.mongodb.insert_posts_batch(posts)
            logger.info(f"Inserted {inserted} Instagram posts into MongoDB")
        
        return posts
    
    def _create_graph_relationships(
        self,
        posts: List[Dict[str, Any]],
        source: str,
    ):
        """Create graph relationships in Neo4j.
        
        Args:
            posts: List of post dictionaries
            source: Platform source
        """
        logger.info(f"Creating graph relationships for {source}")
        
        for post in tqdm(posts, desc="Creating graph nodes"):
            try:
                # Create user node
                if "author" in post:
                    self.neo4j.create_user(
                        user_id=post["author"],
                        source=source,
                        followers=post.get("user", {}).get("followers", 0),
                    )
                
                # Create post node
                post_id = post.get("post_id") or post.get("tweet_id")
                if post_id and post.get("author"):
                    self.neo4j.create_post(
                        post_id=post_id,
                        author_id=post["author"],
                        source=source,
                        created_at=post.get("created_at"),
                        text=post.get("text", post.get("title", "")),
                    )
                
                # Create interaction relationships
                for mention in post.get("mentions", []):
                    if post.get("author"):
                        self.neo4j.create_interaction(
                            from_user_id=post["author"],
                            to_user_id=mention,
                            interaction_type="MENTIONED",
                            post_id=post_id,
                        )
            except Exception as e:
                logger.debug(f"Error creating graph node: {e}")
        
        logger.info("Graph relationships created")
    
    def _generate_report(self, total_samples: int):
        """Generate collection report."""
        mongo_stats = self.mongodb.get_collection_stats()
        neo4j_stats = self.neo4j.get_graph_stats()
        
        report = f"""
# Data Collection Report

Generated: {datetime.now().isoformat()}

## Summary
- Total Samples: {total_samples}
- Target: {self.target_samples}
- Completion: {(total_samples / self.target_samples) * 100:.1f}%

## MongoDB Statistics
- Posts: {mongo_stats['posts']['count']}
- Users: {mongo_stats['users']['count']}
- Comments: {mongo_stats['comments']['count']}

## Neo4j Statistics
- Users: {neo4j_stats['users']}
- Posts: {neo4j_stats['posts']}
- Interactions: {neo4j_stats['interactions']}
- Avg Degree: {neo4j_stats['avg_degree']:.2f}
"""
        
        # Save report
        report_path = Path("data/exports/collection_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w") as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_path}")
        print(report)


def run_pipeline(target_samples: int = 50000):
    """Run the data collection pipeline."""
    pipeline = DataCollectionPipeline(target_samples=target_samples)
    pipeline.run_full_collection()


if __name__ == "__main__":
    run_pipeline()
