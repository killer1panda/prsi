## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.
## 2025-03-07 - Optimization of Live Feed Component
**Learning:** Added `React.memo()` to the `LiveFeedPanel` and `ThreatAnalyzer` components to avoid continuous re-rendering triggered by the parent's `globalScore` interval.
**Action:** Use `React.memo` to memoize heavy child components if their parent changes unrelated state continuously.
