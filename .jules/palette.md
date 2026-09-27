
## 2024-05-18 - Scrollable Region Keyboard Accessibility
**Learning:** Scrollable containers with `overflow: auto` or `overflow: scroll` (e.g. `overflow-y-auto`) are inaccessible to keyboard users unless explicitly made focusable. Keyboard users cannot scroll the container content unless the container itself has focus.
**Action:** Always add `tabIndex={0}`, `role="region"`, `aria-label` and visual focus states (e.g. `outline-none focus-visible:ring-2`) to custom scrollable containers to ensure keyboard navigation functions properly without degrading the mouse experience.
