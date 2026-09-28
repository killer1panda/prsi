## 2024-12-07 - Scrollable Region Accessibility
**Learning:** Custom scrollable containers (like `overflow-y-auto` panels for feeds) need explicit keyboard and screen reader support since they can contain important interactive or live-updating content, and native scrolling with arrows requires the container to be focusable.
**Action:** Always add `tabIndex={0}`, `role="region"`, `aria-label="[Description]"`, and `outline-none focus-visible:ring-2` to `overflow-y-auto` containers to ensure keyboard navigation without breaking mouse UX.
