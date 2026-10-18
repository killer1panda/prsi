## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2026-09-15 - React Next.js Re-render Optimization
**Learning:** Top-level components (e.g., `ThreatIntelligenceDashboard`) that use polling intervals (like `setInterval`) to update a specific piece of state (e.g., `globalScore`) trigger frequent, unnecessary cascading re-renders across all child components. This is a significant performance anti-pattern.
**Action:** Always wrap heavy, state-independent child components (like chart panels or live feeds) with `React.memo()` and explicitly set their `.displayName` property to optimize performance and prevent cascading re-renders from parent polling.
