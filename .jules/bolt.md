## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.
## 2025-03-08 - Unnecessary double-render in Next.js useEffect
**Learning:** In the Next.js `LiveFeedPanel` component, setting the starting state (`sseStatus`) synchronously inside a `useEffect` block on mount triggers an immediate cascading re-render, violating React 18+ strict mode and performance best practices.
**Action:** Always initialize the starting state directly in the `useState` hook (`useState("connecting")`) instead of calling `setState` inside the initial `useEffect` to prevent rapid double-renders and cascading performance anti-patterns.
