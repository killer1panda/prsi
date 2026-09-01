"""
Stateful Complex Event Processing (CEP) Engine for Outrage Cascade Detection.
Evaluates continuous temporal acceleration:
d(Doom)/dt >= θ_accel AND Doom_score >= θ_critical
across a sliding event-time window with out-of-order event tolerance.
"""

import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OutrageCascadeDetectorCEP:
    """
    Complex Event Processing (CEP) engine tracking real-time sliding windows
    per author/topic to trigger early outrage cascade alerts.
    """

    def __init__(
        self,
        velocity_threshold: float = 15.0,  # points per second / hour
        critical_score_threshold: float = 75.0,
        window_duration_sec: float = 10.0,
    ):
        self.velocity_threshold = velocity_threshold
        self.critical_score_threshold = critical_score_threshold
        self.window_duration_sec = window_duration_sec
        # State: {entity_key: [{"timestamp": float, "doom_score": float, "post_id": str}]}
        self.state_buffer: Dict[str, List[Dict[str, Any]]] = {}

    def process_event(
        self,
        entity_key: str,
        post_id: str,
        doom_score: float,
        timestamp_sec: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single streaming event and return an alert dict if a cascade is triggered.
        """
        now = timestamp_sec or time.time()

        if entity_key not in self.state_buffer:
            self.state_buffer[entity_key] = []

        history = self.state_buffer[entity_key]
        history.append({"post_id": post_id, "doom_score": doom_score, "timestamp": now})

        # Evict events older than window_duration_sec
        cutoff = now - self.window_duration_sec
        self.state_buffer[entity_key] = [e for e in history if e["timestamp"] >= cutoff]
        active_window = self.state_buffer[entity_key]

        if len(active_window) >= 3:
            # Sort by timestamp
            active_window.sort(key=lambda x: x["timestamp"])
            first_event = active_window[0]
            latest_event = active_window[-1]

            delta_t = latest_event["timestamp"] - first_event["timestamp"]
            delta_score = latest_event["doom_score"] - first_event["doom_score"]

            if delta_t > 0:
                velocity = delta_score / delta_t
                if (
                    velocity >= self.velocity_threshold
                    and latest_event["doom_score"] >= self.critical_score_threshold
                ):
                    return {
                        "alert_type": "OUTRAGE_CASCADE_SPIKE",
                        "entity_key": entity_key,
                        "velocity_pts_per_sec": round(velocity, 2),
                        "initial_score": first_event["doom_score"],
                        "current_score": latest_event["doom_score"],
                        "events_in_window": len(active_window),
                        "timestamp": now,
                    }

        return None
