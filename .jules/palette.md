
## 2026-10-12 - Keyboard Accessibility for Scrollable Containers
**Learning:** Custom scrollable containers (e.g., `overflow-y-auto`) are ignored by keyboard navigation, making their inner content unscrollable for keyboard-only and screen reader users.
**Action:** Always add `tabIndex={0}`, `role="region"`, and `aria-label` to explicitly label scrollable areas, paired with `focus-visible` styling and `outline-none` to provide accessible keyboard focus states without affecting mouse click styling.
