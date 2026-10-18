
## 2026-09-20 - Accessible Custom Scrollable Containers
**Learning:** Custom scrollable containers (e.g. using 'overflow-y-auto' on standard divs instead of native semantic tags) miss natural keyboard tabbing logic out of the box, which breaks keyboard and screen reader accessibility.
**Action:** Always append 'tabIndex={0}', 'role="region"', an 'aria-label', and a visible keyboard focus state (like 'focus-visible:ring-2') when converting div containers to custom scrollable feeds.
