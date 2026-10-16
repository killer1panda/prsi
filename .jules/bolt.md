## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.
## 2024-10-05 - Optimize Cascading Re-renders in Dashboard
**Learning:** Found an anti-pattern where a root dashboard component running a fast `setInterval` polling hook causes deep, heavy nested elements (like Recharts and dynamic feed panels) to unnecessarily re-render on every tick.
**Action:** Extract heavy, state-independent UI clusters into dedicated components and wrap them in `React.memo()`. Also ensure to initialize fast-changing state natively via `useState` instead of triggering an immediate synchronous setState double-render within a mount `useEffect`.
