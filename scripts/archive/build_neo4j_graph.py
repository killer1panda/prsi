#!/usr/bin/env python3
"""Build Neo4j user interaction graph from processed Reddit data.

Creates:
- User nodes with features (followers, verified, post_count, etc.)
- INTERACTED edges between users (reply, same-thread, same-subreddit)

Usage:
    python build_neo4j_graph.py --data_path data/processed_reddit_multimodal.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.neo4j_connector import get_neo4j

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def create_constraints(neo4j):
    """Create Neo4j constraints and indexes."""
    logger.info("Creating constraints...")

    queries = [
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
        "CREATE INDEX post_id IF NOT EXISTS FOR (p:Post) ON (p.post_id)",
        "CREATE INDEX subreddit_name IF NOT EXISTS FOR (s:Subreddit) ON (s.name)",
    ]

    with neo4j.driver.session(database=neo4j.database) as session:
        for query in queries:
            try:
                session.run(query)
                logger.info(f"  Executed: {query[:50]}...")
            except Exception as e:
                logger.warning(f"  Constraint failed (may already exist): {e}")


def create_user_nodes(neo4j, df: pd.DataFrame, batch_size: int = 1000):
    """Create User nodes from DataFrame."""
    logger.info("Creating user nodes...")

    # Aggregate user features
    user_features = df.groupby('author_id').agg({
        'followers': 'first',
        'verified': 'first',
        'sentiment_polarity': 'mean',
        'toxicity_toxicity': 'mean',
        'text_length': 'count',  # post count
    }).reset_index()

    user_features.columns = ['user_id', 'followers', 'verified', 
                             'avg_sentiment', 'avg_toxicity', 'post_count']
    user_features['verified'] = user_features['verified'].fillna(False).astype(bool)
    user_features['followers'] = user_features['followers'].fillna(0).astype(int)
    user_features['post_count'] = user_features['post_count'].fillna(0).astype(int)
    user_features['avg_sentiment'] = user_features['avg_sentiment'].fillna(0.0)
    user_features['avg_toxicity'] = user_features['avg_toxicity'].fillna(0.0)

    # Batch insert
    query = """
    UNWIND $batch as row
    MERGE (u:User {user_id: row.user_id})
    ON CREATE SET 
        u.followers = row.followers,
        u.verified = row.verified,
        u.post_count = row.post_count,
        u.avg_sentiment = row.avg_sentiment,
        u.avg_toxicity = row.avg_toxicity,
        u.created_at = datetime()
    ON MATCH SET
        u.followers = row.followers,
        u.verified = row.verified,
        u.post_count = row.post_count,
        u.avg_sentiment = row.avg_sentiment,
        u.avg_toxicity = row.avg_toxicity
    """

    with neo4j.driver.session(database=neo4j.database) as session:
        for i in tqdm(range(0, len(user_features), batch_size), desc="Users"):
            batch = user_features.iloc[i:i+batch_size].to_dict('records')
            session.run(query, batch=batch)

    logger.info(f"Created {len(user_features)} user nodes")


def create_post_nodes(neo4j, df: pd.DataFrame, batch_size: int = 1000):
    """Create Post nodes and POSTED relationships."""
    logger.info("Creating post nodes...")

    query = """
    UNWIND $batch as row
    MATCH (u:User {user_id: row.author_id})
    MERGE (p:Post {post_id: row.post_id})
    ON CREATE SET 
        p.text = row.text,
        p.likes = row.likes,
        p.replies = row.replies,
        p.sentiment_polarity = row.sentiment_polarity,
        p.toxicity = row.toxicity,
        p.created_at = row.created_at,
        p.label = row.label
    MERGE (u)-[:POSTED]->(p)
    """

    with neo4j.driver.session(database=neo4j.database) as session:
        for i in tqdm(range(0, len(df), batch_size), desc="Posts"):
            batch = df.iloc[i:i+batch_size].to_dict('records')
            session.run(query, batch=batch)

    logger.info(f"Created {len(df)} post nodes")


def create_subreddit_nodes(neo4j, df: pd.DataFrame):
    """Create Subreddit nodes and POSTED_IN relationships."""
    logger.info("Creating subreddit nodes...")

    subreddits = df['subreddit'].dropna().unique()

    query = """
    UNWIND $names as name
    MERGE (s:Subreddit {name: name})
    """

    with neo4j.driver.session(database=neo4j.database) as session:
        session.run(query, names=subreddits.tolist())

    # Link posts to subreddits
    link_query = """
    UNWIND $batch as row
    MATCH (p:Post {post_id: row.post_id})
    MATCH (s:Subreddit {name: row.subreddit})
    MERGE (p)-[:POSTED_IN]->(s)
    """

    with neo4j.driver.session(database=neo4j.database) as session:
        for i in tqdm(range(0, len(df), 1000), desc="Subreddit links"):
            batch = df[['post_id', 'subreddit']].iloc[i:i+1000].to_dict('records')
            session.run(link_query, batch=batch)

    logger.info(f"Created {len(subreddits)} subreddit nodes")


def create_interaction_edges(neo4j, df: pd.DataFrame):
    """Create INTERACTED edges between users.

    Edge types:
    - Same subreddit (weight=1)
    - Same thread (if thread_id available, weight=2)
    - Reply chain (if parent_id available, weight=3)
    """
    logger.info("Creating interaction edges...")

    # 1. Same subreddit interactions
    logger.info("  Building same-subreddit edges...")

    subreddit_query = """
    MATCH (u1:User)-[:POSTED]->(:Post)-[:POSTED_IN]->(s:Subreddit)<-[:POSTED_IN]-(:Post)<-[:POSTED]-(u2:User)
    WHERE u1.user_id < u2.user_id
    WITH u1, u2, count(s) as shared_subreddits
    MERGE (u1)-[r:INTERACTED]->(u2)
    ON CREATE SET r.weight = shared_subreddits, r.type = 'same_subreddit'
    ON MATCH SET r.weight = r.weight + shared_subreddits
    """

    with neo4j.driver.session(database=neo4j.database) as session:
        result = session.run(subreddit_query)
        summary = result.consume()
        logger.info(f"  Created {summary.counters.relationships_created} subreddit edges")

    # 2. If we have reply information, create reply edges
    if 'parent_id' in df.columns and df['parent_id'].notna().any():
        logger.info("  Building reply chain edges...")

        reply_df = df[df['parent_id'].notna()][['author_id', 'parent_id']].copy()

        # Map parent_id to author
        parent_to_author = df.set_index('post_id')['author_id'].to_dict()
        reply_df['parent_author'] = reply_df['parent_id'].map(parent_to_author)
        reply_df = reply_df[reply_df['parent_author'].notna()]
        reply_df = reply_df[reply_df['author_id'] != reply_df['parent_author']]

        reply_query = """
        UNWIND $batch as row
        MATCH (u1:User {user_id: row.author_id})
        MATCH (u2:User {user_id: row.parent_author})
        WHERE u1.user_id <> u2.user_id
        MERGE (u1)-[r:INTERACTED]->(u2)
        ON CREATE SET r.weight = 3, r.type = 'reply'
        ON MATCH SET r.weight = r.weight + 3
        """

        with neo4j.driver.session(database=neo4j.database) as session:
            for i in tqdm(range(0, len(reply_df), 1000), desc="Reply edges"):
                batch = reply_df.iloc[i:i+1000].to_dict('records')
                session.run(reply_query, batch=batch)

        logger.info(f"  Created reply edges for {len(reply_df)} interactions")

    # 3. Aggregate edge weights
    logger.info("  Normalizing edge weights...")
    normalize_query = """
    MATCH ()-[r:INTERACTED]->()
    WITH max(r.weight) as max_weight
    MATCH ()-[r:INTERACTED]->()
    SET r.normalized_weight = r.weight * 1.0 / max_weight
    """

    with neo4j.driver.session(database=neo4j.database) as session:
        session.run(normalize_query)

    logger.info("Interaction edges complete")


def get_graph_stats(neo4j):
    """Get statistics about the graph."""
    logger.info("Graph statistics:")

    queries = {
        'users': "MATCH (u:User) RETURN count(u) as count",
        'posts': "MATCH (p:Post) RETURN count(p) as count",
        'subreddits': "MATCH (s:Subreddit) RETURN count(s) as count",
        'edges': "MATCH ()-[r:INTERACTED]->() RETURN count(r) as count",
        'avg_degree': "MATCH (u:User)-[r:INTERACTED]-() RETURN avg(count(r)) as avg_degree",
    }

    with neo4j.driver.session(database=neo4j.database) as session:
        for name, query in queries.items():
            try:
                result = session.run(query).single()
                logger.info(f"  {name}: {result['count'] if 'count' in result else result['avg_degree']}")
            except Exception as e:
                logger.warning(f"  Could not get {name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Build Neo4j graph from Reddit data")
    parser.add_argument("--data_path", type=str, default="data/processed_reddit_multimodal.csv")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--clear", action="store_true", help="Clear existing graph before building")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Building Neo4j User Interaction Graph")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} posts")

    # Connect to Neo4j
    neo4j = get_neo4j()

    # Clear if requested
    if args.clear:
        logger.warning("Clearing existing graph...")
        with neo4j.driver.session(database=neo4j.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")

    # Build graph
    create_constraints(neo4j)
    create_user_nodes(neo4j, df, batch_size=args.batch_size)
    create_post_nodes(neo4j, df, batch_size=args.batch_size)
    create_subreddit_nodes(neo4j, df)
    create_interaction_edges(neo4j, df)

    # Stats
    get_graph_stats(neo4j)

    logger.info("=" * 60)
    logger.info("Graph build complete!")
    logger.info("=" * 60)
    logger.info("Next: Run train_multimodal.py to train on this graph")


if __name__ == "__main__":
    main()
