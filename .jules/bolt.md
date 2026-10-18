## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2026-09-29 - Optimization for Next.js polling re-renders
**Learning:** In Next.js applications, top-level components that use polling intervals (like `setInterval` for updating scores) can trigger frequent, unnecessary cascading re-renders of all child components.
**Action:** Always wrap heavy, state-independent child components (e.g., chart panels, live feeds, or analyzers) with `React.memo()` to optimize performance and explicitly set their `.displayName` property to prevent linter warnings.
