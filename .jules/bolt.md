## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2024-05-14 - React.memo effectiveness
**Learning:** In Next.js dashboard components with top-level polling (`setInterval` updating state), child components not wrapped in `React.memo()` re-render unnecessarily on every tick.
**Action:** Wrap heavy UI child components in `React.memo()`. Also, avoid initializing state and immediately overriding it synchronously inside a mount `useEffect()`, as this triggers double renders immediately on mount.
