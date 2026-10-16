## 2024-06-25 - Added keyboard accessibility to scrollable regions
**Learning:** For custom scrollable containers (e.g., `overflow-y-auto`), simply adding the class is not enough. You must add `tabIndex={0}`, `role="region"`, `aria-label`, and `focus-visible` styling (paired with `outline-none`) to ensure keyboard-only and screen reader accessibility without degrading mouse UX.
**Action:** Always verify scrollable regions have keyboard accessibility elements attached, avoiding raw untabbable custom scroll areas.
