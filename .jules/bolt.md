## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.

## 2024-05-18 - Avoid synchronous state updates inside useEffect
**Learning:** Found a component (`LiveFeedPanel`) that was initializing its component state synchronously inside a `useEffect` on mount. This triggers a cascading double-render which hurts performance unnecessarily, and throws a linting warning in Next.js (`react-hooks/set-state-in-effect`).
**Action:** When migrating or writing React components, always initialize the correct starting state directly in the `useState` hook instead of trying to correct the state immediately after the component mounts via `useEffect`.
