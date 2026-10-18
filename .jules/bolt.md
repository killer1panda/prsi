## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.
## 2024-05-14 - Prevent Unnecessary Re-renders in Dashboard
**Learning:** Top-level components (e.g., `ThreatIntelligenceDashboard`) use polling intervals that trigger frequent, unnecessary cascading re-renders.
**Action:** Always wrap heavy, state-independent child components (like chart panels or live feeds) with `React.memo()` to optimize performance. Explicitly set `.displayName` to prevent linter warnings.
## 2024-05-14 - Prevent synchronous setState in useEffect
**Learning:** Initializing starting state directly in the `useState` hook avoids synchronous `setState` in a `useEffect` block on mount, preventing double-renders and performance anti-patterns.
**Action:** Always initialize starting state in `useState` instead of doing it on mount in `useEffect`.
