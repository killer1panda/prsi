## 2024-09-25 - Scroll Container and Textarea Accessibility
**Learning:** Custom scrollable containers (like those using `overflow-y-auto`) must have `tabIndex={0}`, `role="region"`, and `aria-label` alongside `outline-none focus-visible:ring-...` to support keyboard navigation. Also, inputs without visible labels require an `aria-label` for screen readers.
**Action:** Always ensure scrollable areas are focusable and properly styled for focus visibility, and provide ARIA labels for inputs lacking explicit `<label>` tags.
