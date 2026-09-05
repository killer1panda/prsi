## 2024-05-01 - Scrollable Container Accessibility
**Learning:** Custom scrollable containers (using `overflow-y-auto`) are implicitly excluded from the tab order. This creates an accessibility barrier for keyboard-only and screen reader users who cannot focus the container to scroll its contents.
**Action:** Always add `tabIndex={0}`, `role="region"`, an appropriate `aria-label`, and visible focus styles (e.g., `focus-visible:ring`) to custom scrollable containers.
