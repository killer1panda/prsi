
## 2026-10-08 - Accessible Custom Scrollable Containers
**Learning:** Custom scrollable containers (like those using `overflow-y-auto`) are ignored by keyboard navigation unless explicitly made focusable. Without ARIA roles and labels, screen reader users cannot easily understand what the scrolling region contains.
**Action:** Always add `tabIndex={0}`, `role="region"`, an appropriate `aria-label`, and `focus-visible:outline-none focus-visible:ring-2` (or similar) to custom scrollable elements to ensure they are accessible via keyboard and screen readers.
