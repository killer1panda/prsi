## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2026-09-21 - Prevent Cascading Re-renders in Dashboard Components
**Learning:** In top-level Next.js dashboards polling APIs periodically (e.g., `ThreatIntelligenceDashboard` jittering state every 2s), React's default behavior cascades renders down to heavy child components (`LiveFeedPanel` maintaining its own SSE, `ThreatAnalyzer` executing fetch requests) even if their props don't change.
**Action:** Aggressively memoize heavy UI components with `React.memo()` (setting `.displayName` for debugging) when they manage internal state or have stable props, to isolate them from periodic parent re-renders.
