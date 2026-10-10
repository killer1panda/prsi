## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.
## 2025-10-10 - Cascading Renders on mount in useEffect
**Learning:** Initializing state with a loading/connecting value and setting it via `setState` synchronously within a `useEffect` on mount triggers an immediate cascading re-render, creating a performance anti-pattern. This is especially prevalent in data-fetching or streaming components.
**Action:** Always initialize the starting state directly in the `useState` hook (e.g., `useState("connecting")`) rather than setting it synchronously inside a `useEffect` block on mount.

## 2025-10-10 - Unnecessary cascades due to polling intervals
**Learning:** Top-level components containing polling intervals (e.g., `setInterval`) will trigger frequent, unnecessary cascading re-renders down to heavy, pure child components unless they are memoized.
**Action:** Always wrap heavy, pure child components, or components taking stable callbacks (e.g., via `useCallback`), with `React.memo()` in Next.js frontends to optimize performance. Ensure to set `.displayName` on the resulting component for debugging and to avoid linter warnings.
