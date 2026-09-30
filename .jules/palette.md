## 2024-10-18 - [Add accessibility attributes to scrollable containers]
**Learning:** Custom scrollable containers (e.g. overflow-y-auto) require explicit tabIndex=0, role="region", aria-label, and focus-visible outlines to be navigable by keyboard users and screen readers without degrading standard mouse user experience.
**Action:** Always add tabIndex={0}, role="region", aria-label, and proper focus-visible classes with outline-none to any elements using custom scrolling.
