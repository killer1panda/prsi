"""Database connection modules for MongoDB and Neo4j with offline memory fallbacks."""

from datetime import datetime
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, ConnectionFailure, DuplicateKeyError, PyMongoError
from src.config import get_env_var

logger = logging.getLogger(__name__)


class InMemoryCollection:
    """In-memory collection with real dictionary filter matching and cursor support."""

    def __init__(self, name: str):
        self.name = name
        self.docs: List[Dict[str, Any]] = []

    def _matches_query(self, doc: Dict[str, Any], query: Optional[Dict[str, Any]]) -> bool:
        if not query:
            return True
        for k, v in query.items():
            if isinstance(v, dict):
                val = doc.get(k)
                if "$gte" in v and not (val is not None and val >= v["$gte"]):
                    return False
                if "$lte" in v and not (val is not None and val <= v["$lte"]):
                    return False
                if "$gt" in v and not (val is not None and val > v["$gt"]):
                    return False
                if "$lt" in v and not (val is not None and val < v["$lt"]):
                    return False
                if "$ne" in v and val == v["$ne"]:
                    return False
                if "$in" in v and val not in v["$in"]:
                    return False
            else:
                if doc.get(k) != v:
                    return False
        return True

    def insert_one(self, doc: Dict[str, Any]):
        class Result:
            inserted_id = f"mem_{id(doc)}"

        self.docs.append(dict(doc))
        return Result()

    def insert_many(self, docs: List[Dict[str, Any]], ordered=False):
        class Result:
            inserted_ids = [f"mem_{i}_{id(d)}" for i, d in enumerate(docs)]

        self.docs.extend([dict(d) for d in docs])
        return Result()

    def find(self, query: Optional[Dict[str, Any]] = None):
        matched = [d for d in self.docs if self._matches_query(d, query)]

        class Cursor(list):
            def limit(self, n):
                return Cursor(self[:n])

            def sort(self, *args, **kwargs):
                return self

            def skip(self, n):
                return Cursor(self[n:])

            def count(self):
                return len(self)

        return Cursor(matched)

    def find_one(self, query: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        for d in self.docs:
            if self._matches_query(d, query):
                return d
        return None

    def count_documents(self, query: Optional[Dict[str, Any]] = None) -> int:
        return sum(1 for d in self.docs if self._matches_query(d, query))

    def create_index(self, *args, **kwargs):
        pass


class MongoDBConnector:
    """MongoDB connection and operations manager with offline memory fallback."""

    def __init__(
        self,
        uri: Optional[str] = None,
        database: str = "doom_index",
    ):
        self.uri = uri or get_env_var("MONGODB_URI", "mongodb://localhost:27017/doom_index")
        self.database_name = database
        self.is_online = False
        self._fallback_collections = {
            "posts": InMemoryCollection("posts"),
            "users": InMemoryCollection("users"),
            "comments": InMemoryCollection("comments"),
            "cancellation_events": InMemoryCollection("cancellation_events"),
            "meme_templates": InMemoryCollection("meme_templates"),
        }

        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=2000)
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self.is_online = True
            logger.info(f"Connected to MongoDB: {self.database_name}")
            self._setup_collections()
        except (ConnectionFailure, PyMongoError, Exception) as e:
            logger.warning(f"MongoDB offline ({e}); operating in in-memory fallback mode.")
            self.client = None
            self.db = self._fallback_collections
            self.is_online = False

    def _setup_collections(self):
        """Create collections and indexes."""
        if not self.is_online:
            return
        for col_name in ["posts", "users", "comments", "cancellation_events", "meme_templates"]:
            if col_name not in self.db.list_collection_names():
                self.db.create_collection(col_name)

    @property
    def posts(self):
        return self.db["posts"]

    @property
    def users(self):
        return self.db["users"]

    @property
    def comments(self):
        return self.db["comments"]

    @property
    def cancellation_events(self):
        return self.db["cancellation_events"]

    def insert_post(self, post: Dict[str, Any]) -> Optional[str]:
        try:
            post["inserted_at"] = datetime.utcnow()
            result = self.posts.insert_one(post)
            return str(getattr(result, "inserted_id", "mock_id"))
        except DuplicateKeyError:
            logger.warning(f"Duplicate post: {post.get('post_id')}")
            return None
        except PyMongoError as e:
            logger.error(f"PyMongo insert_post error: {e}")
            return None

    def insert_posts_batch(self, posts: List[Dict[str, Any]]) -> int:
        for post in posts:
            post["inserted_at"] = datetime.utcnow()
        try:
            result = self.posts.insert_many(posts, ordered=False)
            return len(getattr(result, "inserted_ids", []))
        except (PyMongoError, BulkWriteError) as e:
            logger.error(f"PyMongo batch insert error: {e}")
            inserted = 0
            for post in posts:
                try:
                    self.posts.insert_one(post)
                    inserted += 1
                except Exception:
                    pass
            return inserted

    def get_posts_by_author(self, author: str, limit: int = 100) -> List[Dict[str, Any]]:
        return list(self.posts.find({"author": author}).limit(limit))

    def get_posts_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        source: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {
            "created_at": {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat(),
            }
        }
        if source:
            query["source"] = source
        return list(self.posts.find(query).limit(limit))

    def get_collection_stats(self) -> Dict[str, Any]:
        stats = {}
        for collection_name in ["posts", "users", "comments", "cancellation_events"]:
            collection = self.db[collection_name]
            stats[collection_name] = {
                "count": collection.count_documents({}),
            }
        return stats

    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


from src.data.neo4j_connector import Neo4jConnector, get_neo4j

# Singleton instances
_mongodb_instance = None


def get_mongodb() -> MongoDBConnector:
    """Get MongoDB connector singleton."""
    global _mongodb_instance
    if _mongodb_instance is None:
        _mongodb_instance = MongoDBConnector()
    return _mongodb_instance
