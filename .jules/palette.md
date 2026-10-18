
## 2024-11-20 - Accessible Scrollable Regions
**Learning:** Custom scrollable containers (like `overflow-y-auto` panels for feeds) are inaccessible to keyboard and screen reader users by default.
**Action:** Always add `tabIndex={0}`, `role="region"`, an `aria-label`, and visible focus styles (like `focus-visible:ring-1`) to standalone scrollable elements so they can be navigated natively.
