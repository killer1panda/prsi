## 2023-10-24 - Accessible Scrollable Containers
**Learning:** Custom scrollable containers (like `overflow-y-auto` divs used in Live Feeds) are often inaccessible to keyboard users and screen readers, trapping them or preventing them from scrolling content.
**Action:** Always add `tabIndex={0}`, `role="region"`, a descriptive `aria-label`, and `focus-visible` styling (with `outline-none`) to custom scroll containers to ensure keyboard and screen reader accessibility without degrading mouse UX.
