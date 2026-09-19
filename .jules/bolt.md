## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.
## 2025-03-08 - Preventing cascading re-renders in Next.js from polling intervals
**Learning:** Top-level components (like `ThreatIntelligenceDashboard`) use polling intervals (e.g., `setInterval`) to update global state like scores or timestamps. This triggers frequent, unnecessary cascading re-renders for heavy, state-independent child components (like chart panels or live feeds).
**Action:** Always wrap heavy, state-independent child components with `React.memo()` and explicitly set their `.displayName` property to optimize performance and prevent rapid double-renders in React components.
