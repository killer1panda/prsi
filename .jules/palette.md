## 2025-10-05 - Custom Scrollable Regions Accessibility
**Learning:** Custom scrollable containers (e.g., using `overflow-y-auto`) natively break keyboard accessibility because they can't be tabbed into, making it impossible for keyboard users to scroll without the mouse.
**Action:** Always add `tabIndex={0}`, `role="region"`, `aria-label`, and `focus-visible` styling (with `outline-none`) to `overflow-y-auto` elements so they receive focus safely for screen readers and keyboard users alike.
