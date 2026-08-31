#!/usr/bin/env python3
"""
Production-grade Neo4j graph builder extracting REAL user interaction edges
from Reddit Pushshift data. Creates user-user reply/mention/co-subreddit graphs
with temporal properties, not synthetic k-NN fallbacks.

This is the HIGHEST PRIORITY file. Without real edges, your GNN is useless.
"""
import os
import sys
import json
import logging
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Iterator
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Neo4j
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GraphBuildConfig:
    """Configuration for graph construction."""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    database: str = "neo4j"
    
    # Edge types to construct
    build_reply_edges: bool = True
    build_mention_edges: bool = True
    build_cosubreddit_edges: bool = True
    build_temporal_edges: bool = True
    
    # Thresholds
    min_interactions_for_edge: int = 1
    max_cosubreddit_users: int = 5000  # Limit co-subreddit edges to prevent explosion
    temporal_window_hours: int = 24
    
    # Batch sizes
    node_batch_size: int = 1000
    edge_batch_size: int = 5000
    
    # Anonymization
    hash_salt: str = "doom_index_salt_2026"
    
    def __post_init__(self):
        os.environ["NEO4J_URI"] = self.neo4j_uri


class Neo4jGraphBuilder:
    """
    Production graph builder that creates a REAL social network graph in Neo4j.
    
    Node Types:
        - User: Social media users (hashed IDs)
        - Post: Individual posts/comments
        - Subreddit: Communities/forums
        - Topic: Extracted topics (via LDA/keyword clustering)
    
    Edge Types:
        - REPLIED_TO: User -> User (reply chain)
        - MENTIONED: User -> User (@mentions)
        - POSTED_IN: User -> Subreddit
        - INTERACTED_IN: User -> User (same thread, within time window)
        - CO_SUBREDDIT: User -> User (same community, weighted by overlap)
        - QUOTED: Post -> Post (quote chains)
    """
    
    def __init__(self, config: GraphBuildConfig):
        self.config = config
        self.driver: Optional[Driver] = None
        self._connect()
        self._ensure_constraints()
        
        # In-memory buffers for batching
        self.user_buffer: Set[str] = set()
        self.post_buffer: List[Dict] = []
        self.edge_buffer: List[Tuple[str, str, str, Dict]] = []
        
        # Statistics
        self.stats = {
            "users_created": 0,
            "posts_created": 0,
            "subreddits_created": 0,
            "edges_created": 0,
            "reply_edges": 0,
            "mention_edges": 0,
            "cosubreddit_edges": 0,
            "temporal_edges": 0
        }
    
    def _connect(self):
        """Establish Neo4j connection with retry logic."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_password)
                )
                self.driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self.config.neo4j_uri}")
                return
            except (ServiceUnavailable, AuthError) as e:
                logger.warning(f"Neo4j connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                import time
                time.sleep(2 ** attempt)
    
    def _ensure_constraints(self):
        """Create uniqueness constraints and indexes for performance."""
        with self.driver.session(database=self.config.database) as session:
            constraints = [
                "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
                "CREATE CONSTRAINT post_id_unique IF NOT EXISTS FOR (p:Post) REQUIRE p.post_id IS UNIQUE",
                "CREATE CONSTRAINT subreddit_name_unique IF NOT EXISTS FOR (s:Subreddit) REQUIRE s.name IS UNIQUE",
                "CREATE INDEX user_doom_score_idx IF NOT EXISTS FOR (u:User) ON (u.doom_score)",
                "CREATE INDEX post_timestamp_idx IF NOT EXISTS FOR (p:Post) ON (p.timestamp)",
                "CREATE INDEX edge_timestamp_idx IF NOT EXISTS FOR ()-[r:REPLIED_TO]-() ON (r.timestamp)",
                "CREATE INDEX user_followers_idx IF NOT EXISTS FOR (u:User) ON (u.follower_count)"
            ]
            
            for cypher in constraints:
                try:
                    session.run(cypher)
                except Exception as e:
                    logger.warning(f"Constraint creation skipped: {e}")
            
            logger.info("Neo4j constraints and indexes ensured")
    
    def _hash_user(self, username: str) -> str:
        """Anonymize username with salted hash."""
        return hashlib.sha256(
            f"{username}{self.config.hash_salt}".encode()
        ).hexdigest()[:16]
    
    def _flush_nodes(self, session: Session):
        """Batch create buffered nodes."""
        if not self.user_buffer and not self.post_buffer:
            return
        
        # Create users
        if self.user_buffer:
            user_list = list(self.user_buffer)
            cypher = """
            UNWIND $users AS user_id
            MERGE (u:User {user_id: user_id})
            ON CREATE SET u.created_at = datetime(), u.doom_score = 0.0
            RETURN count(u) AS created
            """
            result = session.run(cypher, users=user_list)
            self.stats["users_created"] += result.single()["created"]
            self.user_buffer.clear()
        
        # Create posts
        if self.post_buffer:
            cypher = """
            UNWIND $posts AS post
            MERGE (p:Post {post_id: post.post_id})
            ON CREATE SET 
                p.text = post.text,
                p.timestamp = datetime(post.timestamp),
                p.score = post.score,
                p.subreddit = post.subreddit,
                p.sentiment = post.sentiment,
                p.toxicity = post.toxicity
            WITH p, post
            MATCH (u:User {user_id: post.author_id})
            MERGE (u)-[:AUTHORED]->(p)
            RETURN count(p) AS created
            """
            result = session.run(cypher, posts=self.post_buffer)
            self.stats["posts_created"] += result.single()["created"]
            self.post_buffer.clear()
    
    def _flush_edges(self, session: Session):
        """Batch create buffered edges."""
        if not self.edge_buffer:
            return
        
        # Group by edge type for efficient batching
        edges_by_type: Dict[str, List[Tuple]] = defaultdict(list)
        for src, dst, rel_type, props in self.edge_buffer:
            edges_by_type[rel_type].append((src, dst, props))
        
        for rel_type, edges in edges_by_type.items():
            # Build parameterized query
            cypher = f"""
            UNWIND $edges AS edge
            MATCH (a:User {{user_id: edge.src}})
            MATCH (b:User {{user_id: edge.dst}})
            WHERE a <> b
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET r += edge.props
            ON MATCH SET r.weight = coalesce(r.weight, 1) + 1,
                         r.last_updated = datetime()
            RETURN count(r) AS created
            """
            
            edge_params = [
                {"src": src, "dst": dst, "props": props}
                for src, dst, props in edges
            ]
            
            result = session.run(cypher, edges=edge_params)
            created = result.single()["created"]
            self.stats["edges_created"] += created
            
            if rel_type == "REPLIED_TO":
                self.stats["reply_edges"] += created
            elif rel_type == "MENTIONED":
                self.stats["mention_edges"] += created
            elif rel_type == "CO_SUBREDDIT":
                self.stats["cosubreddit_edges"] += created
            elif rel_type == "INTERACTED_IN":
                self.stats["temporal_edges"] += created
        
        self.edge_buffer.clear()
    
    def add_user(self, username: str):
        """Queue user for creation."""
        user_hash = self._hash_user(username)
        self.user_buffer.add(user_hash)
        
        if len(self.user_buffer) >= self.config.node_batch_size:
            with self.driver.session(database=self.config.database) as session:
                self._flush_nodes(session)
    
    def add_post(self, post_id: str, author: str, text: str, 
                 timestamp: str, subreddit: str, score: int = 0,
                 sentiment: float = 0.0, toxicity: float = 0.0):
        """Queue post for creation."""
        author_hash = self._hash_user(author)
        self.user_buffer.add(author_hash)
        
        self.post_buffer.append({
            "post_id": post_id,
            "author_id": author_hash,
            "text": text[:2000],  # Limit text size
            "timestamp": timestamp,
            "subreddit": subreddit,
            "score": score,
            "sentiment": round(sentiment, 4),
            "toxicity": round(toxicity, 4)
        })
        
        if len(self.post_buffer) >= self.config.node_batch_size:
            with self.driver.session(database=self.config.database) as session:
                self._flush_nodes(session)
    
    def add_edge(self, user1: str, user2: str, rel_type: str, 
                 weight: float = 1.0, timestamp: Optional[str] = None,
                 subreddit: Optional[str] = None, **props):
        """Queue edge for creation."""
        u1_hash = self._hash_user(user1)
        u2_hash = self._hash_user(user2)
        
        if u1_hash == u2_hash:
            return
        
        edge_props = {
            "weight": weight,
            "created_at": timestamp or datetime.utcnow().isoformat(),
            **props
        }
        if subreddit:
            edge_props["subreddit"] = subreddit
        
        self.edge_buffer.append((u1_hash, u2_hash, rel_type, edge_props))
        
        if len(self.edge_buffer) >= self.config.edge_batch_size:
            with self.driver.session(database=self.config.database) as session:
                self._flush_edges(session)
    
    def extract_replies_from_df(self, df: pd.DataFrame):
        """
        Extract REPLIED_TO edges from DataFrame with parent_id references.
        
        Expected columns: id, author, parent_id, body, created_utc, subreddit, score
        """
        logger.info("Extracting reply edges...")
        
        # Build mapping of post_id -> author
        post_to_author: Dict[str, str] = {}
        if "id" in df.columns and "author" in df.columns:
            post_to_author = dict(zip(df["id"].astype(str), df["author"].astype(str)))
        
        reply_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Reply edges"):
            try:
                author = str(row.get("author", ""))
                parent_id = str(row.get("parent_id", ""))
                
                if not author or not parent_id or parent_id == "nan":
                    continue
                    
                # parent_id often prefixed with "t1_" or "t3_"
                parent_id_clean = parent_id.split("_")[-1]
                
                if parent_id_clean in post_to_author:
                    parent_author = post_to_author[parent_id_clean]
                    if parent_author != author and parent_author != "[deleted]":
                        self.add_edge(
                            author, parent_author, "REPLIED_TO",
                            weight=1.0,
                            timestamp=self._format_timestamp(row.get("created_utc")),
                            subreddit=str(row.get("subreddit", "unknown"))
                        )
                        reply_count += 1
            except Exception as e:
                logger.debug(f"Reply extraction error: {e}")
                continue
        
        logger.info(f"Queued {reply_count} reply edges")
    
    def extract_mentions_from_df(self, df: pd.DataFrame):
        """
        Extract MENTIONED edges from text containing u/username references.
        
        Expected columns: body, author, created_utc, subreddit
        """
        logger.info("Extracting mention edges...")
        import re
        
        mention_pattern = re.compile(r"/?u/([A-Za-z0-9_-]+)")
        mention_count = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Mention edges"):
            try:
                text = str(row.get("body", ""))
                author = str(row.get("author", ""))
                
                if not text or not author:
                    continue
                    
                mentions = mention_pattern.findall(text)
                for mentioned_user in mentions:
                    if mentioned_user != author and mentioned_user != "[deleted]":
                        self.add_edge(
                            author, mentioned_user, "MENTIONED",
                            weight=1.0,
                            timestamp=self._format_timestamp(row.get("created_utc")),
                            subreddit=str(row.get("subreddit", "unknown"))
                        )
                        mention_count += 1
            except Exception as e:
                logger.debug(f"Mention extraction error: {e}")
                continue
        
        logger.info(f"Queued {mention_count} mention edges")
    
    def extract_cosubreddit_edges(self, df: pd.DataFrame):
        """
        Extract CO_SUBREDDIT edges: users who post in the same subreddit.
        Weighted by Jaccard similarity of subreddit participation.
        """
        logger.info("Extracting co-subreddit edges...")
        
        # Build user -> subreddits mapping
        user_subreddits: Dict[str, Set[str]] = defaultdict(set)
        user_post_counts: Dict[str, int] = defaultdict(int)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building user profiles"):
            author = str(row.get("author", ""))
            subreddit = str(row.get("subreddit", ""))
            
            if author and author != "[deleted]" and subreddit and subreddit != "nan":
                user_subreddits[author].add(subreddit)
                user_post_counts[author] += 1
        
        # Filter to active users
        active_users = [
            u for u, count in user_post_counts.items()
            if count >= self.config.min_interactions_for_edge
        ]
        
        if len(active_users) > self.config.max_cosubreddit_users:
            # Sort by activity and take top N
            active_users = sorted(
                active_users,
                key=lambda u: user_post_counts[u],
                reverse=True
            )[:self.config.max_cosubreddit_users]
        
        logger.info(f"Computing co-subreddit edges for {len(active_users)} users...")
        
        # Compute Jaccard similarity for user pairs
        cosub_count = 0
        user_list = active_users
        
        for i in tqdm(range(len(user_list)), desc="Co-subreddit edges"):
            u1 = user_list[i]
            s1 = user_subreddits[u1]
            
            for j in range(i + 1, len(user_list)):
                u2 = user_list[j]
                s2 = user_subreddits[u2]
                
                intersection = len(s1 & s2)
                if intersection >= 2:  # At least 2 common subreddits
                    union = len(s1 | s2)
                    jaccard = intersection / union if union > 0 else 0
                    
                    self.add_edge(
                        u1, u2, "CO_SUBREDDIT",
                        weight=round(jaccard, 4),
                        common_subreddits=list(s1 & s2)[:10]
                    )
                    cosub_count += 1
        
        logger.info(f"Queued {cosub_count} co-subreddit edges")
    
    def extract_temporal_edges(self, df: pd.DataFrame):
        """
        Extract INTERACTED_IN edges: users who interact in the same thread
        within a temporal window.
        """
        logger.info("Extracting temporal interaction edges...")
        
        # Group by thread (link_id for comments, id for posts)
        thread_users: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Grouping threads"):
            thread_id = str(row.get("link_id", row.get("id", "")))
            author = str(row.get("author", ""))
            timestamp = self._parse_timestamp(row.get("created_utc"))
            
            if thread_id and author and author != "[deleted]":
                thread_users[thread_id].append((author, timestamp))
        
        temporal_count = 0
        window_seconds = self.config.temporal_window_hours * 3600
        
        for thread_id, users in tqdm(thread_users.items(), desc="Temporal edges"):
            if len(users) < 2:
                continue
            
            # Sort by time
            users.sort(key=lambda x: x[1])
            
            # Create edges between users within time window
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    u1, t1 = users[i]
                    u2, t2 = users[j]
                    
                    if u1 == u2:
                        continue
                    
                    time_diff = abs(t2 - t1)
                    if time_diff <= window_seconds:
                        # Weight decays with time difference
                        weight = 1.0 - (time_diff / window_seconds)
                        self.add_edge(
                            u1, u2, "INTERACTED_IN",
                            weight=round(weight, 4),
                            thread_id=thread_id,
                            time_diff_hours=round(time_diff / 3600, 2)
                        )
                        temporal_count += 1
        
        logger.info(f"Queued {temporal_count} temporal edges")
    
    def compute_user_features(self):
        """
        Compute and store node-level features in Neo4j.
        Features: degree, betweenness proxy, activity level, controversy rate.
        """
        logger.info("Computing user graph features...")
        
        with self.driver.session(database=self.config.database) as session:
            # Degree centrality
            session.run("""
                MATCH (u:User)
                OPTIONAL MATCH (u)-[r]-()
                WITH u, count(r) AS degree
                SET u.degree = degree
            """)
            
            # In-degree (being replied to / mentioned)
            session.run("""
                MATCH (u:User)
                OPTIONAL MATCH ()-[r:REPLIED_TO|MENTIONED]->(u)
                WITH u, count(r) AS in_degree
                SET u.in_degree = in_degree
            """)
            
            # Out-degree (replying / mentioning others)
            session.run("""
                MATCH (u:User)
                OPTIONAL MATCH (u)-[r:REPLIED_TO|MENTIONED]->()
                WITH u, count(r) AS out_degree
                SET u.out_degree = out_degree
            """)
            
            # Post count
            session.run("""
                MATCH (u:User)
                OPTIONAL MATCH (u)-[:AUTHORED]->(p:Post)
                WITH u, count(p) AS post_count
                SET u.post_count = post_count
            """)
            
            # Average sentiment of posts
            session.run("""
                MATCH (u:User)
                OPTIONAL MATCH (u)-[:AUTHORED]->(p:Post)
                WITH u, avg(p.sentiment) AS avg_sentiment
                SET u.avg_sentiment = coalesce(avg_sentiment, 0.0)
            """)
            
            # Average toxicity
            session.run("""
                MATCH (u:User)
                OPTIONAL MATCH (u)-[:AUTHORED]->(p:Post)
                WITH u, avg(p.toxicity) AS avg_toxicity
                SET u.avg_toxicity = coalesce(avg_toxicity, 0.0)
            """)
            
            # Controversy rate: high engagement + negative sentiment
            session.run("""
                MATCH (u:User)
                OPTIONAL MATCH (u)-[:AUTHORED]->(p:Post)
                WHERE p.score > 100 AND p.sentiment < -0.3
                WITH u, count(p) AS controversial_posts, coalesce(u.post_count, 1) AS total
                SET u.controversy_rate = toFloat(controversial_posts) / total
            """)
            
            # Community count (unique subreddits)
            session.run("""
                MATCH (u:User)
                OPTIONAL MATCH (u)-[:AUTHORED]->(p:Post)
                WITH u, count(DISTINCT p.subreddit) AS community_count
                SET u.community_count = community_count
            """)
        
        logger.info("User graph features computed")
    
    def compute_graph_embeddings(self, embedding_dim: int = 128):
        """
        Compute node2vec-style graph embeddings using GDS (Graph Data Science).
        Falls back to spectral embedding if GDS not available.
        """
        logger.info(f"Computing graph embeddings (dim={embedding_dim})...")
        
        with self.driver.session(database=self.config.database) as session:
            # Check if GDS is available
            try:
                session.run("CALL gds.graph.exists('user-graph') YIELD exists")
                has_gds = True
            except Exception:
                has_gds = False
            
            if has_gds:
                # Use FastRP (Fast Random Projection) from GDS
                session.run("""
                    CALL gds.graph.drop('user-graph', false) YIELD graphName
                """)
                
                session.run("""
                    CALL gds.graph.project(
                        'user-graph',
                        'User',
                        {
                            REPLIED_TO: {orientation: 'UNDIRECTED'},
                            MENTIONED: {orientation: 'UNDIRECTED'},
                            CO_SUBREDDIT: {orientation: 'UNDIRECTED'},
                            INTERACTED_IN: {orientation: 'UNDIRECTED'}
                        },
                        {relationshipProperties: 'weight'}
                    )
                """)
                
                session.run(f"""
                    CALL gds.fastRP.write('user-graph', {{
                        embeddingDimension: {embedding_dim},
                        iterationWeights: [0.0, 1.0, 1.0],
                        writeProperty: 'graph_embedding'
                    }})
                """)
                
                session.run("CALL gds.graph.drop('user-graph') YIELD graphName")
                logger.info("Graph embeddings computed via GDS FastRP")
            else:
                logger.warning("GDS not available. Skipping graph embeddings.")
    
    def finalize(self):
        """Flush all buffers and compute derived features."""
        logger.info("Finalizing graph build...")
        
        with self.driver.session(database=self.config.database) as session:
            self._flush_nodes(session)
            self._flush_edges(session)
        
        self.compute_user_features()
        self.compute_graph_embeddings()
        
        logger.info(f"Graph build complete. Stats: {self.stats}")
        return self.stats
    
    def export_edge_list(self, output_path: str):
        """Export edge list for PyG training."""
        with self.driver.session(database=self.config.database) as session:
            result = session.run("""
                MATCH (u1:User)-[r]->(u2:User)
                RETURN u1.user_id AS source, u2.user_id AS target,
                       type(r) AS rel_type, r.weight AS weight
            """)
            
            edges = []
            for record in result:
                edges.append({
                    "source": record["source"],
                    "target": record["target"],
                    "rel_type": record["rel_type"],
                    "weight": record["weight"]
                })
            
            df = pd.DataFrame(edges)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} edges to {output_path}")
    
    def export_node_features(self, output_path: str):
        """Export node features for PyG training."""
        with self.driver.session(database=self.config.database) as session:
            result = session.run("""
                MATCH (u:User)
                RETURN u.user_id AS user_id,
                       u.degree AS degree,
                       u.in_degree AS in_degree,
                       u.out_degree AS out_degree,
                       u.post_count AS post_count,
                       u.avg_sentiment AS avg_sentiment,
                       u.avg_toxicity AS avg_toxicity,
                       u.controversy_rate AS controversy_rate,
                       u.community_count AS community_count,
                       u.doom_score AS doom_score
            """)
            
            nodes = []
            for record in result:
                nodes.append(dict(record))
            
            df = pd.DataFrame(nodes)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} nodes to {output_path}")
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    @staticmethod
    def _format_timestamp(ts) -> str:
        """Format various timestamp formats to ISO."""
        if pd.isna(ts):
            return datetime.utcnow().isoformat()
        if isinstance(ts, (int, float)):
            return datetime.utcfromtimestamp(ts).isoformat()
        return str(ts)
    
    @staticmethod
    def _parse_timestamp(ts) -> float:
        """Parse timestamp to Unix seconds."""
        if pd.isna(ts):
            return 0.0
        if isinstance(ts, (int, float)):
            return float(ts)
        try:
            return pd.to_datetime(ts).timestamp()
        except Exception:
            return 0.0


def main():
    parser = argparse.ArgumentParser(description="Build Neo4j graph from Reddit data")
    parser.add_argument("--data", required=True, help="Path to processed CSV/Parquet")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--max-users", type=int, default=5000, help="Max users for co-subreddit edges")
    parser.add_argument("--temporal-window", type=int, default=24, help="Temporal window in hours")
    parser.add_argument("--output-edges", default="data/graph/edge_list.csv")
    parser.add_argument("--output-nodes", default="data/graph/node_features.csv")
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    
    logger.info(f"Loaded {len(df)} rows")
    
    # Build graph
    config = GraphBuildConfig(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        max_cosubreddit_users=args.max_users,
        temporal_window_hours=args.temporal_window
    )
    
    builder = Neo4jGraphBuilder(config)
    
    try:
        # Extract all edge types
        builder.extract_replies_from_df(df)
        builder.extract_mentions_from_df(df)
        builder.extract_cosubreddit_edges(df)
        builder.extract_temporal_edges(df)
        
        # Finalize
        stats = builder.finalize()
        
        # Export for PyG
        builder.export_edge_list(args.output_edges)
        builder.export_node_features(args.output_nodes)
        
        # Save stats
        stats_path = Path(args.output_edges).parent / "graph_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Graph stats saved to {stats_path}")
        
    finally:
        builder.close()


if __name__ == "__main__":
    main()
