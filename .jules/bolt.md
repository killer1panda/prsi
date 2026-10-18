## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2023-10-27 - Optimizing unnecessary re-renders in Next.js
**Learning:** Top-level components (e.g., `ThreatIntelligenceDashboard`) in the Next.js frontend (`apps/web`) use polling intervals that trigger frequent, unnecessary cascading re-renders for all child components, even those independent of the polling state.
**Action:** Always wrap heavy, state-independent child components (like chart panels or live feeds, e.g., `LiveFeedPanel`, `ThreatAnalyzer`) with `React.memo()` to optimize performance and set `.displayName` to prevent linter warnings.
