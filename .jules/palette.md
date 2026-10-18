
## 2024-09-24 - Accessible Scrollable Regions
**Learning:** Custom scrollable containers (like `overflow-y-auto` divs used for feeds/lists) are inaccessible to keyboard-only users because they cannot scroll them without a focusable element inside.
**Action:** Always add `tabIndex={0}`, `role="region"`, a descriptive `aria-label`, and `focus-visible` styling (paired with `outline-none`) to `overflow` containers to ensure keyboard-only and screen reader accessibility without degrading mouse UX.
