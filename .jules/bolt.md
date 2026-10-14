## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2024-10-14 - Initialize State Correctly to Prevent Double Renders
**Learning:** Found a performance anti-pattern where a component initialized its state asynchronously via `useEffect` triggering a cascading re-render (and lint warning). The state should be initialized directly within the hook call if it is static and synchronous.
**Action:** When inspecting component lifecycle states, ensure synchronous state values are defined inline at the `useState` declaration rather than pushed into a mount `useEffect`.
