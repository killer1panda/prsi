## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.
## 2026-09-20 - Prevent React state init inside useEffect
**Learning:** Setting a starting state inside `useEffect` immediately on mount (like `setSseStatus('connecting')`) causes an unnecessary cascading re-render that blocks the main thread and harms performance. The linter flagged this correctly.
**Action:** Initialize starting states directly in the `useState` hook instead.
