## 2024-03-24 - Make Scrollable Containers Keyboard Accessible
**Learning:** Custom scrollable containers (like those using `overflow-y-auto`) are not keyboard-accessible by default. Keyboard-only users cannot scroll the content unless they can focus the container itself.
**Action:** Always add `tabIndex={0}`, `role="region"`, and an `aria-label` to custom scrollable containers, along with `focus-visible` styling, to ensure they can be focused and announced properly by screen readers.## 2024-10-15 - React useEffect setState refactor
**Learning:** Initializing state with a loading or default value inside `useEffect` can cause an unnecessary re-render and trigger ESLint errors (e.g. `react-hooks/set-state-in-effect`).
**Action:** Always prefer setting the initial state directly in the `useState` definition when possible to avoid flash-of-unstyled-content or brief invalid states.
