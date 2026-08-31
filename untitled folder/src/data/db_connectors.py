"""Database connection modules for MongoDB and Neo4j."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, DuplicateKeyError

from src.config import get_env_var


class MongoDBConnector:
    """MongoDB connection and operations manager."""
    
    def __init__(
        self,
        uri: str = None,
        database: str = "doom_index",
    ):
        """Initialize MongoDB connection.
        
        Args:
            uri: MongoDB connection URI
            database: Database name
        """
        self.uri = uri or get_env_var("MONGODB_URI", "mongodb://localhost:27017/doom_index")
        self.database_name = database
        
        # Initialize client
        self.client = MongoClient(self.uri)
        self.db = self.client[self.database_name]
        
        # Test connection
        try:
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self.database_name}")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
        
        # Initialize collections
        self._setup_collections()
    
    def _setup_collections(self):
        """Create collections and indexes."""
        # Posts collection
        if "posts" not in self.db.list_collection_names():
            self.db.create_collection("posts")
        
        posts_collection = self.db["posts"]
        posts_collection.create_index([("post_id", ASCENDING)], unique=True)
        posts_collection.create_index([("source", ASCENDING)])
        posts_collection.create_index([("created_at", DESCENDING)])
        posts_collection.create_index([("author", ASCENDING)])
        
        # Users collection
        if "users" not in self.db.list_collection_names():
            self.db.create_collection("users")
        
        users_collection = self.db["users"]
        users_collection.create_index([("user_id", ASCENDING)], unique=True)
        users_collection.create_index([("source", ASCENDING)])
        
        # Comments collection
        if "comments" not in self.db.list_collection_names():
            self.db.create_collection("comments")
        
        comments_collection = self.db["comments"]
        comments_collection.create_index([("comment_id", ASCENDING)], unique=True)
        comments_collection.create_index([("post_id", ASCENDING)])
        comments_collection.create_index([("author", ASCENDING)])
        
        # Cancellation events collection
        if "cancellation_events" not in self.db.list_collection_names():
            self.db.create_collection("cancellation_events")
        
        events_collection = self.db["cancellation_events"]
        events_collection.create_index([("event_id", ASCENDING)], unique=True)
        events_collection.create_index([("date", DESCENDING)])
        
        logger.info("MongoDB collections and indexes created")
    
    @property
    def posts(self) -> Collection:
        """Get posts collection."""
        return self.db["posts"]
    
    @property
    def users(self) -> Collection:
        """Get users collection."""
        return self.db["users"]
    
    @property
    def comments(self) -> Collection:
        """Get comments collection."""
        return self.db["comments"]
    
    @property
    def cancellation_events(self) -> Collection:
        """Get cancellation events collection."""
        return self.db["cancellation_events"]
    
    def insert_post(self, post: Dict[str, Any]) -> str:
        """Insert a post document.
        
        Args:
            post: Post data dictionary
            
        Returns:
            Inserted document ID
        """
        try:
            post["inserted_at"] = datetime.utcnow()
            result = self.posts.insert_one(post)
            return str(result.inserted_id)
        except DuplicateKeyError:
            logger.warning(f"Duplicate post: {post.get('post_id')}")
            return None
    
    def insert_posts_batch(self, posts: List[Dict[str, Any]]) -> int:
        """Insert multiple posts.
        
        Args:
            posts: List of post dictionaries
            
        Returns:
            Number of inserted documents
        """
        for post in posts:
            post["inserted_at"] = datetime.utcnow()
        
        try:
            result = self.posts.insert_many(posts, ordered=False)
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Batch insert error: {e}")
            # Try inserting one by one
            inserted = 0
            for post in posts:
                try:
                    self.posts.insert_one(post)
                    inserted += 1
                except DuplicateKeyError:
                    pass
            return inserted
    
    def get_posts_by_author(
        self,
        author: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get posts by author.
        
        Args:
            author: Author ID (hashed)
            limit: Maximum results
            
        Returns:
            List of posts
        """
        return list(self.posts.find({"author": author}).limit(limit))
    
    def get_posts_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        source: str = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Get posts within date range.
        
        Args:
            start_date: Start date
            end_date: End date
            source: Filter by source (twitter, reddit, instagram)
            limit: Maximum results
            
        Returns:
            List of posts
        """
        query = {
            "created_at": {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat(),
            }
        }
        
        if source:
            query["source"] = source
        
        return list(self.posts.find(query).limit(limit))
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections.
        
        Returns:
            Dictionary with collection statistics
        """
        stats = {}
        for collection_name in ["posts", "users", "comments", "cancellation_events"]:
            collection = self.db[collection_name]
            stats[collection_name] = {
                "count": collection.count_documents({}),
            }
        return stats
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")


# Singleton instance
_mongodb_instance = None


def get_mongodb() -> MongoDBConnector:
    """Get MongoDB connector singleton."""
    global _mongodb_instance
    if _mongodb_instance is None:
        _mongodb_instance = MongoDBConnector()
    return _mongodb_instance
