## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## $(date +%Y-%m-%d) - Prevent cascading re-renders in ThreatIntelligenceDashboard
**Learning:** In the Next.js frontend (`apps/web`), top-level components using polling intervals can trigger frequent, unnecessary cascading re-renders across the entire component tree. Chart panels and heavy components like `LiveFeedPanel` were re-rendering every 2 seconds despite their internal state not depending on the parent's polling.
**Action:** Extract inline complex UI components (like Recharts graphs) into separate functional components and explicitly wrap them and other state-independent children with `React.memo()` to block unnecessary render propagation.
