## 2024-05-18 - Scrollable Container Accessibility
**Learning:** Custom scrollable containers (like those using `overflow-y-auto`) are inaccessible to keyboard-only and screen reader users by default. Without a `tabIndex`, users cannot focus the container to scroll it with arrow keys, and without a `role` and `aria-label`, screen readers won't announce it properly.
**Action:** Always add `tabIndex={0}`, `role="region"`, `aria-label`, and `focus-visible` styling to custom scrollable containers to ensure full accessibility.
