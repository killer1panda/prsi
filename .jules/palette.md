## 2024-05-24 - Accessible Scrollable Regions
**Learning:** Custom scrollable containers (like `overflow-y-auto` panels in custom React components) are completely inaccessible to keyboard users unless explicitly made focusable, which prevents users from reading overflowing content without a mouse.
**Action:** When adding `overflow-auto` or `overflow-y-auto` to a container, always pair it with `tabIndex={0}`, `role="region"`, `aria-label`, and `focus-visible` outline styles so it can be focused and scrolled via keyboard without breaking visual styling.
