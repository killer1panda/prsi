## 2024-10-10 - Custom Scrollable Container Accessibility
**Learning:** Custom scrollable containers (like `overflow-y-auto` divs) are completely unreachable for keyboard and screen reader users by default, which is a major accessibility issue for components like the Live Feed Panel in this app.
**Action:** Always add `tabIndex={0}`, `role="region"`, a descriptive `aria-label`, and `focus-visible` styling (paired with `outline-none`) to ensure keyboard-only and screen reader accessibility without degrading mouse UX on custom scroll containers.
