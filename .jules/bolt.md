## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2026-09-17 - Unnecessary cascading re-renders in Next.js parent components
**Learning:** In the Next.js frontend (`apps/web`), top-level components like `ThreatIntelligenceDashboard` use polling intervals (e.g., `setInterval`) that trigger state updates. This causes frequent, unnecessary cascading re-renders of all child components, even those whose state hasn't changed.
**Action:** Always wrap heavy, state-independent child components (like `LiveFeedPanel` or `ThreatAnalyzer`) with `React.memo()` to prevent these re-renders and optimize performance. Remember to explicitly set `.displayName` on memoized anonymous components to prevent linter warnings.
