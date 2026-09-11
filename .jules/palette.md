## $(date +%Y-%m-%d) - Custom Scrollable Container Accessibility
**Learning:** Custom scrollable containers (using `overflow-y-auto` or similar) are inaccessible to keyboard and screen reader users without additional markup. They cannot be scrolled using the keyboard and are ignored by screen readers.
**Action:** Always add `tabIndex={0}`, an appropriate `role` (like `"region"`), an `aria-label`, and `focus-visible` styling to ensure custom scrollable regions can receive focus and be accessed via keyboard and assistive technologies.
