## 2024-10-09 - Accessible Scrollable Regions
**Learning:** Adding accessibility to dynamically populated scrollable regions (like live feeds) requires careful pairing of `role="region"`, `aria-label`, and `tabIndex={0}` alongside explicitly defined custom visual focus outlines (`outline-none focus-visible:ring-2 ...`) so they don't break the keyboard UX.
**Action:** When creating `.overflow-y-auto` elements specifically used for large lists or continuous scroll, always equip them with semantic region attributes and visible focus indicators.
