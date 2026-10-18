## 2026-09-13 - Scrollable Container Keyboard Accessibility
**Learning:** Custom scrollable containers (like those using `overflow-y-auto`) are not keyboard focusable by default, meaning users navigating via keyboard cannot scroll through them. Screen readers also may not announce their purpose.
**Action:** Always add `tabIndex={0}`, `role="region"`, an appropriate `aria-label`, and `focus-visible` styling to `overflow-y-auto` containers to ensure keyboard scrollability and proper screen reader semantics.
