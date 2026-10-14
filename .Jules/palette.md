## 2024-03-24 - Make Scrollable Containers Keyboard Accessible
**Learning:** Custom scrollable containers (like those using `overflow-y-auto`) are not keyboard-accessible by default. Keyboard-only users cannot scroll the content unless they can focus the container itself.
**Action:** Always add `tabIndex={0}`, `role="region"`, and an `aria-label` to custom scrollable containers, along with `focus-visible` styling, to ensure they can be focused and announced properly by screen readers.## 2024-10-14 - React State Anti-Pattern in Effects
**Learning:** Found a component (`LiveFeedPanel`) where the initial "connecting" state was set synchronously inside a `useEffect` on mount, triggering cascading renders and violating React's strict mode recommendations.
**Action:** When inspecting components for UX fixes, check for rapid double-renders on mount. Initialize the intended starting state directly in the `useState` hook rather than setting it inside a mount effect.
