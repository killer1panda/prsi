## 2024-10-24 - Accessible minimalist forms with aria-label

**Learning:** The minimalist form design in `eleved.html` sacrificed screen reader accessibility by omitting labels in favor of placeholder text. When creating custom CSS classes (like `sr-only`) is against constraints, adding `aria-label` directly to form elements is the proper way to restore accessibility without compromising the design.
**Action:** Always add `aria-label` to form inputs when the design system removes explicit labels and custom CSS is not allowed.
