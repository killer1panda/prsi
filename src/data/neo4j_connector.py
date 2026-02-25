"""Neo4j graph database connector module."""

from typing import Any, Dict, List, Optional

from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from src.config import get_env_var


class Neo4jConnector:
    """Neo4j connection and operations manager."""
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = "neo4j",
    ):
        """Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Database name
        """
        self.uri = uri or get_env_var("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or get_env_var("NEO4J_USER", "neo4j")
        self.password = password or get_env_var("NEO4J_PASSWORD")
        
        if not self.password:
            raise ValueError("NEO4J_PASSWORD environment variable must be set")
        
        self.database = database
        
        # Initialize driver
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )
        
        # Test connection
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j: {self.database}")
        except ServiceUnavailable as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise
        
        # Setup constraints
        self._setup_constraints()
    
    def _setup_constraints(self):
        """Create constraints and indexes."""
        with self.driver.session(database=self.database) as session:
            # User constraints
            try:
                session.run("""
                    CREATE CONSTRAINT user_id_unique IF NOT EXISTS
                    FOR (u:User) REQUIRE u.user_id IS UNIQUE
                """)
            except Exception as e:
                logger.debug(f"User constraint may already exist: {e}")
            
            # Post constraints
            try:
                session.run("""
                    CREATE CONSTRAINT post_id_unique IF NOT EXISTS
                    FOR (p:Post) REQUIRE p.post_id IS UNIQUE
                """)
            except Exception as e:
                logger.debug(f"Post constraint may already exist: {e}")
            
            # Create indexes
            session.run("CREATE INDEX user_source_idx IF NOT EXISTS FOR (u:User) ON (u.source)")
            session.run("CREATE INDEX post_created_idx IF NOT EXISTS FOR (p:Post) ON (p.created_at)")
            session.run("CREATE INDEX post_source_idx IF NOT EXISTS FOR (p:Post) ON (p.source)")
            
            logger.info("Neo4j constraints and indexes created")
    
    def create_user(
        self,
        user_id: str,
        source: str,
        username: str = None,
        followers: int = 0,
        verified: bool = False,
        **kwargs,
    ) -> bool:
        """Create or update a user node.
        
        Args:
            user_id: Unique user identifier (hashed)
            source: Platform source (twitter, reddit, instagram)
            username: Original username (optional)
            followers: Follower count
            verified: Verification status
            **kwargs: Additional user properties
            
        Returns:
            True if successful
        """
        with self.driver.session(database=self.database) as session:
            query = """
                MERGE (u:User {user_id: $user_id})
                SET u.source = $source,
                    u.followers = $followers,
                    u.verified = $verified,
                    u.updated_at = datetime()
                SET u += $properties
                RETURN u
            """
            
            result = session.run(
                query,
                user_id=user_id,
                source=source,
                followers=followers,
                verified=verified,
                properties=kwargs,
            )
            
            return result.single() is not None
    
    def create_post(
        self,
        post_id: str,
        author_id: str,
        source: str,
        created_at: str = None,
        text: str = None,
        **kwargs,
    ) -> bool:
        """Create a post node and link to author.
        
        Args:
            post_id: Unique post identifier
            author_id: Author user ID
            source: Platform source
            created_at: Post creation timestamp
            text: Post text content
            **kwargs: Additional post properties
            
        Returns:
            True if successful
        """
        with self.driver.session(database=self.database) as session:
            query = """
                MERGE (p:Post {post_id: $post_id})
                SET p.source = $source,
                    p.created_at = $created_at,
                    p.text = $text,
                    p.updated_at = datetime()
                SET p += $properties
                WITH p
                MATCH (u:User {user_id: $author_id})
                MERGE (u)-[:POSTED]->(p)
                RETURN p
            """
            
            result = session.run(
                query,
                post_id=post_id,
                author_id=author_id,
                source=source,
                created_at=created_at,
                text=text,
                properties=kwargs,
            )
            
            return result.single() is not None
    
    def create_interaction(
        self,
        from_user_id: str,
        to_user_id: str,
        interaction_type: str,
        post_id: str = None,
        weight: float = 1.0,
    ) -> bool:
        """Create an interaction relationship between users.
        
        Args:
            from_user_id: Source user ID
            to_user_id: Target user ID
            interaction_type: Type (REPLIED, MENTIONED, RETWEETED, QUOTED)
            post_id: Related post ID
            weight: Interaction weight
            
        Returns:
            True if successful
        """
        with self.driver.session(database=self.database) as session:
            query = """
                MATCH (from:User {user_id: $from_user_id})
                MATCH (to:User {user_id: $to_user_id})
                MERGE (from)-[r:INTERACTED {type: $interaction_type}]->(to)
                SET r.weight = r.weight + $weight,
                    r.updated_at = datetime()
                WITH r
                OPTIONAL MATCH (p:Post {post_id: $post_id})
                FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (from)-[:INTERACTED_WITH_POST {type: $interaction_type}]->(p)
                )
                RETURN r
            """
            
            result = session.run(
                query,
                from_user_id=from_user_id,
                to_user_id=to_user_id,
                interaction_type=interaction_type,
                post_id=post_id,
                weight=weight,
            )
            
            return result.single() is not None
    
    def get_user_network(
        self,
        user_id: str,
        depth: int = 2,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get user's interaction network.
        
        Args:
            user_id: Target user ID
            depth: Network depth (1 = direct connections)
            limit: Maximum nodes to return
            
        Returns:
            Dictionary with nodes and edges
        """
        with self.driver.session(database=self.database) as session:
            query = f"""
                MATCH path = (u:User {{user_id: $user_id}})-[:INTERACTED*1..{depth}]-(connected:User)
                RETURN u, connected, relationships(path) as rels
                LIMIT {limit}
            """
            
            result = session.run(query, user_id=user_id)
            
            nodes = {}
            edges = []
            
            for record in result:
                # Add user node
                user = dict(record["u"])
                nodes[user["user_id"]] = user
                
                # Add connected node
                connected = dict(record["connected"])
                nodes[connected["user_id"]] = connected
                
                # Add edges
                for rel in record["rels"]:
                    edges.append({
                        "source": rel.start_node["user_id"],
                        "target": rel.end_node["user_id"],
                        "type": rel["type"],
                        "weight": rel.get("weight", 1.0),
                    })
            
            return {
                "nodes": list(nodes.values()),
                "edges": edges,
            }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        with self.driver.session(database=self.database) as session:
            # Node counts
            user_count = session.run("MATCH (u:User) RETURN count(u) as count").single()["count"]
            post_count = session.run("MATCH (p:Post) RETURN count(p) as count").single()["count"]
            
            # Relationship counts
            interaction_count = session.run("""
                MATCH ()-[r:INTERACTED]->()
                RETURN count(r) as count
            """).single()["count"]
            
            # Average degree
            avg_degree_result = session.run("""
                MATCH (u:User)-[r:INTERACTED]-()
                WITH u, count(r) as degree
                RETURN avg(degree) as avg_degree
            """).single()
            avg_degree = avg_degree_result["avg_degree"] if avg_degree_result else 0
            
            return {
                "users": user_count,
                "posts": post_count,
                "interactions": interaction_count,
                "avg_degree": avg_degree or 0,
            }
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
        logger.info("Neo4j connection closed")


# Singleton instance
_neo4j_instance = None


def get_neo4j() -> Neo4jConnector:
    """Get Neo4j connector singleton."""
    global _neo4j_instance
    if _neo4j_instance is None:
        _neo4j_instance = Neo4jConnector()
    return _neo4j_instance
