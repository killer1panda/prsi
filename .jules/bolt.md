## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2026-10-08 - React.memo for components with independent state intervals
**Learning:** In Next.js frontend (`apps/web`), top-level components with frequent state updates (like polling intervals for `globalScore`) cause unnecessary cascading re-renders across all child components, even those with their own independent state logic (like SSE feeds or text analysis).
**Action:** Wrap heavy, state-independent child components with `React.memo()` to prevent these costly re-renders. Explicitly set `.displayName` to avoid linter warnings.
