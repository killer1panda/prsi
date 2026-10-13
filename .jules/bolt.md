## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.
## 2026-10-13 - [Synchronous State Update in useEffect Leads to Cascading Renders]
**Learning:** When adding initial states that require immediate DOM sync, avoid setting state synchronously inside `useEffect` immediately on mount (e.g. `setSseStatus('connecting')`), as this causes an immediate cascading double-render and fails strict linting rules. The state should just be initialized correctly in `useState()` instead.
**Action:** Audit and correct `useState` initial values rather than trying to boot up state within a mount effect.
