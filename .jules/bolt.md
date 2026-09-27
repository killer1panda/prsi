## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2024-05-18 - Prevent Cascading Re-renders from Polling Intervals in Next.js
**Learning:** In Next.js frontend applications, top-level components (e.g., `ThreatIntelligenceDashboard`) that use polling intervals (`setInterval`) trigger frequent, unnecessary cascading re-renders of all child components.
**Action:** Always wrap heavy, state-independent child components (like `LiveFeedPanel`, `LiveScoreDisplay`, and `ThreatAnalyzer`) with `React.memo()` to block these unnecessary cascading re-renders and explicitly set `.displayName` to prevent linter warnings.
