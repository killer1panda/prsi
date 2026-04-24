"""
Production Feature Store for Doom Index.
Manages online (low-latency) and offline (training) feature consistency
with versioning, materialization, and point-in-time correctness.
"""
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import redis

logger = logging.getLogger(__name__)


@dataclass
class FeatureView:
    """Definition of a feature view (logical grouping of features)."""
    name: str
    entities: List[str]  # e.g., ["user_id", "post_id"]
    features: List[str]
    ttl: int = 86400  # Time-to-live in seconds
    online: bool = True
    description: str = ""
    version: str = "1.0.0"
    owner: str = "doom-team"
    tags: Dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

    def feature_hash(self) -> str:
        """Generate hash of feature definition for versioning."""
        content = json.dumps({
            "name": self.name,
            "entities": sorted(self.entities),
            "features": sorted(self.features),
            "version": self.version
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class FeatureValue:
    """A single feature value with metadata."""
    value: Any
    timestamp: datetime
    feature_view: str
    feature_name: str
    entity_key: str


class FeatureStore:
    """
    Simple production feature store with Redis online store and Parquet offline store.
    Ensures training/serving consistency via feature versioning.
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379,
                 redis_db: int = 0, offline_path: str = "data/feature_store"):
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db,
            decode_responses=False, socket_connect_timeout=5
        )
        self.offline_path = Path(offline_path)
        self.offline_path.mkdir(parents=True, exist_ok=True)

        self.feature_views: Dict[str, FeatureView] = {}
        self._entity_registry: Dict[str, Dict] = defaultdict(dict)

        logger.info(f"FeatureStore initialized: redis={redis_host}:{redis_port}, offline={offline_path}")

    def register_feature_view(self, view: FeatureView):
        """Register a new feature view."""
        self.feature_views[view.name] = view
        logger.info(f"Registered feature view: {view.name} v{view.version}")

    def _make_key(self, entity_type: str, entity_id: str, feature_view: str, 
                  feature_name: str) -> str:
        """Construct Redis key for online store."""
        return f"fs:{feature_view}:{entity_type}:{entity_id}:{feature_name}"

    def push_online(self, entity_type: str, entity_id: str, 
                    feature_view: str, features: Dict[str, Any],
                    timestamp: Optional[datetime] = None):
        """
        Push features to online store (Redis) for low-latency serving.

        Args:
            entity_type: e.g., "user", "post"
            entity_id: entity identifier
            feature_view: registered feature view name
            features: Dict of feature_name -> value
            timestamp: observation timestamp
        """
        ts = timestamp or datetime.utcnow()
        ts_str = ts.isoformat()
        view = self.feature_views.get(feature_view)
        ttl = view.ttl if view else 86400

        pipe = self.redis_client.pipeline()
        for feat_name, value in features.items():
            key = self._make_key(entity_type, entity_id, feature_view, feat_name)
            payload = {
                "value": self._serialize(value),
                "timestamp": ts_str,
                "version": view.version if view else "unknown"
            }
            pipe.setex(key, ttl, json.dumps(payload))

        pipe.execute()
        logger.debug(f"Pushed {len(features)} features to online store for {entity_type}:{entity_id}")

    def get_online(self, entity_type: str, entity_id: str,
                   feature_view: str, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fetch features from online store.

        Returns:
            Dict of feature_name -> value (stale features return None)
        """
        view = self.feature_views.get(feature_view)
        if feature_names is None and view:
            feature_names = view.features
        elif feature_names is None:
            feature_names = []

        pipe = self.redis_client.pipeline()
        keys = []
        for feat_name in feature_names:
            key = self._make_key(entity_type, entity_id, feature_view, feat_name)
            keys.append((feat_name, key))
            pipe.get(key)

        results = pipe.execute()
        output = {}
        for (feat_name, _), raw in zip(keys, results):
            if raw is None:
                output[feat_name] = None
            else:
                payload = json.loads(raw)
                output[feat_name] = self._deserialize(payload["value"])

        return output

    def write_offline(self, df: pd.DataFrame, feature_view: str,
                      partition_date: Optional[str] = None):
        """
        Write feature dataframe to offline store (Parquet) for training.
        Maintains point-in-time correctness via partitioning.

        Args:
            df: DataFrame with entity columns + feature columns + timestamp
            feature_view: feature view name
            partition_date: YYYY-MM-DD partition (default: today)
        """
        date = partition_date or datetime.utcnow().strftime("%Y-%m-%d")
        view_dir = self.offline_path / feature_view / f"date={date}"
        view_dir.mkdir(parents=True, exist_ok=True)

        # Write with metadata
        view = self.feature_views.get(feature_view)
        metadata = {
            "feature_view": feature_view,
            "version": view.version if view else "unknown",
            "partition": date,
            "num_rows": len(df),
            "columns": list(df.columns),
            "written_at": datetime.utcnow().isoformat()
        }

        parquet_path = view_dir / "features.parquet"
        df.to_parquet(parquet_path, index=False)

        meta_path = view_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Wrote {len(df)} rows to offline store: {parquet_path}")

    def read_offline(self, feature_view: str, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     entity_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Read features from offline store with time filtering.

        Args:
            feature_view: feature view name
            start_date: YYYY-MM-DD inclusive
            end_date: YYYY-MM-DD inclusive
            entity_ids: filter by specific entities
        Returns:
            Combined DataFrame
        """
        view_dir = self.offline_path / feature_view
        if not view_dir.exists():
            return pd.DataFrame()

        dfs = []
        for partition_dir in sorted(view_dir.glob("date=*")):
            date_str = partition_dir.name.replace("date=", "")
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue

            parquet_path = partition_dir / "features.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                if entity_ids is not None and "entity_id" in df.columns:
                    df = df[df["entity_id"].isin(entity_ids)]
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def materialize(self, feature_view: str, entity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Point-in-time correct join: get features as they existed at entity timestamps.
        Critical for preventing training/serving skew.

        Args:
            feature_view: feature view name
            entity_df: DataFrame with [entity_id, event_timestamp] columns
        Returns:
            DataFrame with joined features
        """
        # Read all offline data for this view
        all_features = self.read_offline(feature_view)
        if all_features.empty:
            return entity_df

        # Ensure timestamp columns exist
        if "event_timestamp" not in entity_df.columns:
            raise ValueError("entity_df must have 'event_timestamp' column")
        if "timestamp" not in all_features.columns:
            raise ValueError("Feature data must have 'timestamp' column")

        entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])
        all_features["timestamp"] = pd.to_datetime(all_features["timestamp"])

        # As-of merge: get most recent features before event_timestamp
        result = pd.merge_asof(
            entity_df.sort_values("event_timestamp"),
            all_features.sort_values("timestamp"),
            left_on="event_timestamp",
            right_on="timestamp",
            by="entity_id",
            direction="backward"
        )

        return result

    def get_feature_vector(self, entity_type: str, entity_id: str,
                           feature_views: List[str]) -> Dict[str, Any]:
        """Get complete feature vector from multiple views for online inference."""
        vector = {}
        for view_name in feature_views:
            features = self.get_online(entity_type, entity_id, view_name)
            vector.update(features)
        return vector

    def _serialize(self, value: Any) -> Any:
        """Serialize feature value for storage."""
        if isinstance(value, (np.ndarray, torch.Tensor)):
            return value.tolist()
        return value

    def _deserialize(self, value: Any) -> Any:
        """Deserialize feature value."""
        return value

    def health_check(self) -> Dict[str, Any]:
        """Check feature store health."""
        try:
            self.redis_client.ping()
            redis_ok = True
        except Exception as e:
            redis_ok = False
            logger.error(f"Redis health check failed: {e}")

        return {
            "redis_online": redis_ok,
            "offline_path_exists": self.offline_path.exists(),
            "registered_views": list(self.feature_views.keys()),
            "status": "healthy" if redis_ok else "degraded"
        }
