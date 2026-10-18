## 2025-03-07 - Optimization in temporal edge extraction nested loop
**Learning:** In Neo4j graph production builds (`extract_temporal_edges`), calculating user interactions within a timeframe used an $O(N^2)$ nested loop despite the users list already being sorted by timestamp. This is a common anti-pattern in temporal aggregations.
**Action:** Always check if loops iterating over sorted data (like timestamps) can be short-circuited with an early `break` condition (e.g., breaking once the `time_diff` exceeds the window) to reduce the algorithm's actual time complexity.
## 2025-03-08 - React.memo Component Wrapping and Callbacks
**Learning:** Using `React.memo` on a child component to prevent re-renders when a parent component is updating state (e.g. from an interval) works well for components with primitive props, but fails if the child takes a callback prop that is recreated on every parent render. The parent must use `useCallback` for the passed prop to ensure referential stability and make `React.memo` effective.
**Action:** When applying `React.memo` for performance optimization, always trace callback props back to their parent and ensure they are wrapped in `useCallback` to prevent breaking the shallow equality check.
## 2025-03-08 - Asynchronous State Updates inside useEffect
**Learning:** Initializing a state value then synchronously updating it via `setState` within a `useEffect` hook on mount causes a cascading double render.
**Action:** When a component is supposed to start in a certain state (like "connecting" for a websocket/SSE), initialize the state to that value directly in the `useState` hook rather than relying on an immediate `useEffect` update, resulting in fewer re-renders.
