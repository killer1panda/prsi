## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2026-10-06 - Optimization of cascading renders in React
**Learning:** The `ThreatIntelligenceDashboard` relies on a 2-second interval, leading to cascading, performance-heavy re-renders on components like `LiveFeedPanel` and `ThreatAnalyzer`. Also found a useEffect that updated state synchronously on mount.
**Action:** When working on rapidly updating Next.js dashboards, use `React.memo` aggressively on child components to block inherited re-renders from parent loops, and set initial states inside the hook instead of immediately in `useEffect`.
