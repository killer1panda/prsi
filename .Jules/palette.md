## 2023-10-27 - Scrollable Region Accessibility
**Learning:** Found that custom `overflow-y-auto` elements (like the Inference Output Stream container) trap keyboard users if they do not have a focusable state, meaning screen reader and keyboard-only users cannot scroll the internal content.
**Action:** Always add `tabIndex={0}`, `role="region"`, `aria-label`, and `focus-visible` styling to custom scrollable containers to ensure they are keyboard accessible and structurally semantic.
