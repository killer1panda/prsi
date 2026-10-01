## 2023-10-01 - Scrollable Container Keyboard Accessibility

**Learning:** Custom scrollable containers (using `overflow-y-auto` or `overflow-x-auto`) are inaccessible to keyboard-only users by default. Screen readers and keyboard navigation can skip over these areas, preventing users from seeing content that extends beyond the visible area.

**Action:** Always add `tabIndex={0}`, `role="region"`, `aria-label`, and `focus-visible:outline-none focus-visible:ring-*` styling to custom scrollable containers to ensure they are fully accessible to keyboard and screen reader users without degrading mouse user experience.
