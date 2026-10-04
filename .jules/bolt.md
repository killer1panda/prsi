## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2024-05-18 - Optimize Next.js top-level components with intervals
**Learning:** In the Next.js frontend (`apps/web`), top-level components (like `ThreatIntelligenceDashboard`) that use polling intervals trigger frequent, unnecessary cascading re-renders across all child components. This is a significant anti-pattern for performance, particularly for heavy child components.
**Action:** Always wrap heavy, state-independent child components (like chart panels or live feeds) with `React.memo()` when the parent component has frequent state updates (like intervals or active polling) to prevent them from unnecessarily re-rendering. Set the `.displayName` on the wrapped components to avoid linter warnings.
