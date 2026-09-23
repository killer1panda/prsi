
## 2024-05-18 - Scrollable Container Accessibility
**Learning:** Custom scrollable containers (e.g., using `overflow-y-auto`) are not keyboard accessible by default and screen readers might not announce them properly.
**Action:** Always add `tabIndex={0}`, `role="region"`, `aria-label`, and `focus-visible` styling (paired with `outline-none`) to custom scrollable containers to ensure keyboard-only and screen reader accessibility without degrading mouse UX.
