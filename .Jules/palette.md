## 2024-03-24 - Make Scrollable Containers Keyboard Accessible
**Learning:** Custom scrollable containers (like those using `overflow-y-auto`) are not keyboard-accessible by default. Keyboard-only users cannot scroll the content unless they can focus the container itself.
**Action:** Always add `tabIndex={0}`, `role="region"`, and an `aria-label` to custom scrollable containers, along with `focus-visible` styling, to ensure they can be focused and announced properly by screen readers.
## 2023-10-25 - Scrollable Container and Form Element Accessibility Focus
**Learning:** Custom scrollable containers (like `overflow-y-auto` elements) need explicit `tabIndex={0}`, `role="region"`, and `aria-label` to be accessible for keyboard and screen reader navigation. Additionally, `textarea` form elements that lack attached text labels must contain an explicit `aria-label` to be adequately interpreted.
**Action:** Always verify scrollable components are accessible via focus tracking natively, particularly in standard interface cards and panels without pre-built keyboard logic. Ensure raw textareas are assigned `aria-label` properties when they omit standard `<label>` tags.
