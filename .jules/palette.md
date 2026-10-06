## 2024-05-24 - Accessible Scrollable Regions
**Learning:** Custom scrollable containers (e.g., using `overflow-y-auto`) need explicit accessibility attributes so keyboard-only users can scroll them and screen readers can identify them.
**Action:** Always add `tabIndex={0}`, `role="region"`, `aria-label`, and `focus-visible` styling (paired with `outline-none`) to `overflow-y-auto` containers.
