## 2024-03-24 - Make Scrollable Containers Keyboard Accessible
**Learning:** Custom scrollable containers (like those using `overflow-y-auto`) are not keyboard-accessible by default. Keyboard-only users cannot scroll the content unless they can focus the container itself.
**Action:** Always add `tabIndex={0}`, `role="region"`, and an `aria-label` to custom scrollable containers, along with `focus-visible` styling, to ensure they can be focused and announced properly by screen readers.

## 2026-09-17 - Focus indicators for scrollable regions
**Learning:** When making overflow-y-auto regions accessible with tabIndex={0} and role='region', standard focus rings often don't match the design system visually or conflict with mouse interactions. Using focus-visible styles (like focus-visible:ring-1 focus-visible:ring-rose-500) paired with outline-none creates a much cleaner UX that only highlights for keyboard users.
**Action:** Always pair tabIndex={0} on scrollable regions with explicit focus-visible utility classes and outline-none to ensure accessible keyboard navigation without degrading mouse UX.
