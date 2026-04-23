"""Graph extraction from Neo4j to PyTorch Geometric.

Extracts user-interaction graphs with node features for GraphSAGE training.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.data.neo4j_connector import get_neo4j

logger = logging.getLogger(__name__)


class GraphExtractor:
    """Extract graph data from Neo4j for PyG."""

    def __init__(self, neo4j=None):
        self.neo4j = neo4j or get_neo4j()

    def extract_user_graph(
        self,
        min_interactions: int = 1,
        max_users: int = 50000
    ) -> Tuple[Data, pd.DataFrame]:
        """Extract user-interaction graph from Neo4j.

        Returns:
            pyg_data: PyTorch Geometric Data object
            user_df: DataFrame with user_id mapping and features
        """
        logger.info("Extracting user graph from Neo4j...")

        # 1. Get all users with features
        users = self._get_user_features(max_users)
        user_df = pd.DataFrame(users)

        if len(user_df) == 0:
            logger.warning("No users found in Neo4j. Returning empty graph.")
            return self._create_empty_graph(), user_df

        # Create user_id → index mapping
        user_ids = user_df['user_id'].tolist()
        user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

        # 2. Get interaction edges
        edges = self._get_interaction_edges(min_interactions)

        # Filter edges to only include users in our set
        edge_list = []
        edge_weights = []

        for edge in edges:
            src = edge['from_user']
            dst = edge['to_user']
            if src in user_id_to_idx and dst in user_id_to_idx:
                edge_list.append([user_id_to_idx[src], user_id_to_idx[dst]])
                edge_weights.append(edge.get('weight', 1.0))

        if len(edge_list) == 0:
            logger.warning("No edges found. Creating k-NN fallback graph.")
            edge_index = self._knn_fallback(user_df)
            edge_weight = None
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        # Make undirected (interactions are bidirectional in nature)
        edge_index = to_undirected(edge_index)

        # 3. Build node feature matrix
        x = self._build_node_features(user_df)

        # 4. Build labels (doom score from user activity patterns)
        y = self._build_node_labels(user_df)

        # 5. Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=len(user_df)
        )

        if edge_weight is not None:
            data.edge_weight = edge_weight

        # Store mapping for later lookup
        data.user_id_map = user_id_to_idx
        data.user_ids = user_ids

        logger.info(f"Graph extracted: {data.num_nodes} nodes, {data.num_edges} edges")
        logger.info(f"Node features: {data.num_node_features}, Labels: {data.y.shape}")

        return data, user_df

    def _get_user_features(self, max_users: int) -> List[Dict]:
        """Query Neo4j for user nodes with computed features."""
        query = """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
        WITH u, 
             count(p) as post_count,
             avg(p.sentiment_polarity) as avg_sentiment,
             avg(p.toxicity) as avg_toxicity,
             sum(CASE WHEN p.is_controversial = true THEN 1 ELSE 0 END) as controversy_count
        RETURN u.user_id as user_id,
               u.followers as followers,
               u.verified as verified,
               post_count,
               avg_sentiment,
               avg_toxicity,
               controversy_count,
               CASE WHEN post_count > 0 THEN controversy_count * 1.0 / post_count ELSE 0 END as controversy_rate
        LIMIT $max_users
        """

        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run(query, max_users=max_users)
            users = []
            for record in result:
                users.append({
                    'user_id': record['user_id'],
                    'followers': record['followers'] or 0,
                    'verified': 1.0 if record['verified'] else 0.0,
                    'post_count': record['post_count'] or 0,
                    'avg_sentiment': record['avg_sentiment'] or 0.0,
                    'avg_toxicity': record['avg_toxicity'] or 0.0,
                    'controversy_count': record['controversy_count'] or 0,
                    'controversy_rate': record['controversy_rate'] or 0.0,
                })
            return users

    def _get_interaction_edges(self, min_interactions: int) -> List[Dict]:
        """Query Neo4j for user-user interactions."""
        query = """
        MATCH (u1:User)-[r:INTERACTED]->(u2:User)
        WHERE u1.user_id <> u2.user_id
        WITH u1.user_id as from_user, u2.user_id as to_user, 
             sum(r.weight) as total_weight, count(r) as interaction_count
        WHERE interaction_count >= $min_interactions
        RETURN from_user, to_user, total_weight as weight, interaction_count
        """

        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run(query, min_interactions=min_interactions)
            edges = []
            for record in result:
                edges.append({
                    'from_user': record['from_user'],
                    'to_user': record['to_user'],
                    'weight': record['weight'],
                    'interaction_count': record['interaction_count']
                })
            return edges

    def _build_node_features(self, user_df: pd.DataFrame) -> torch.Tensor:
        """Build normalized node feature matrix."""
        feature_cols = [
            'followers', 'verified', 'post_count', 
            'avg_sentiment', 'avg_toxicity', 'controversy_rate'
        ]

        features = user_df[feature_cols].fillna(0).values.astype(np.float32)

        # Log-transform followers (highly skewed)
        features[:, 0] = np.log1p(features[:, 0])

        # Normalize
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features = (features - mean) / std

        return torch.tensor(features, dtype=torch.float)

    def _build_node_labels(self, user_df: pd.DataFrame) -> torch.Tensor:
        """Build node labels based on user cancellation risk.

        Heuristic: High controversy rate + negative sentiment + high toxicity = at-risk
        """
        controversy = user_df['controversy_rate'].fillna(0).values
        sentiment = user_df['avg_sentiment'].fillna(0).values
        toxicity = user_df['avg_toxicity'].fillna(0).values

        # Risk score: higher controversy, more negative sentiment, higher toxicity
        risk_score = (
            controversy * 3.0 +
            (-sentiment) * 2.0 +  # Negative sentiment increases risk
            toxicity * 2.0
        )

        # Binary label: top 20% are "high risk"
        threshold = np.percentile(risk_score, 80)
        labels = (risk_score >= threshold).astype(np.int64)

        return torch.tensor(labels, dtype=torch.long)

    def _knn_fallback(self, user_df: pd.DataFrame, k: int = 5) -> torch.Tensor:
        """Create k-NN graph from feature similarity if no edges exist."""
        from sklearn.neighbors import kneighbors_graph

        features = user_df[['followers', 'verified', 'post_count', 
                           'avg_sentiment', 'avg_toxicity', 'controversy_rate']].fillna(0).values

        adj = kneighbors_graph(features, n_neighbors=min(k, len(features)-1), 
                               mode='connectivity', include_self=False)

        # Convert to edge_index
        coo = adj.tocoo()
        edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)

        return edge_index

    def _create_empty_graph(self) -> Data:
        """Create minimal graph when Neo4j is empty."""
        return Data(x=torch.zeros((1, 6)), edge_index=torch.zeros((2, 0), dtype=torch.long))

    def get_user_embedding(
        self,
        user_id: str,
        model,
        data: Data
    ) -> Optional[torch.Tensor]:
        """Get GraphSAGE embedding for a specific user."""
        if not hasattr(data, 'user_id_map') or user_id not in data.user_id_map:
            return None

        idx = data.user_id_map[user_id]
        model.eval()
        with torch.no_grad():
            embeddings = model.get_embeddings(data.x, data.edge_index)
            return embeddings[idx]


if __name__ == "__main__":
    extractor = GraphExtractor()
    data, df = extractor.extract_user_graph()
    print(f"Graph: {data}")
    print(df.head())
