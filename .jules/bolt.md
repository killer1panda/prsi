## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.
## 2024-12-05 - Optimize Next.js Top-Level Polling Cascades
**Learning:** Polling intervals (like `setInterval`) in top-level Next.js/React components (e.g., `ThreatIntelligenceDashboard`) cause frequent and unnecessary cascading re-renders across all child components (like chart panels or live feeds), even those whose props are independent of the interval's state.
**Action:** Always proactively wrap state-independent or heavily rendered child components with `React.memo()` and explicitly set their `.displayName` to eliminate wasted render cycles caused by top-level state changes.
