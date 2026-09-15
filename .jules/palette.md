
## 2024-05-24 - Accessible Custom Scrollable Containers
**Learning:** Custom scrollable containers (like `overflow-y-auto` divs) are not focusable by default, rendering their content inaccessible to keyboard-only and screen reader users if they cannot be reached.
**Action:** Always add `tabIndex={0}`, `role="region"`, `aria-label`, and `focus-visible` ring styling to custom scrollable containers to ensure keyboard-only and screen reader accessibility.
