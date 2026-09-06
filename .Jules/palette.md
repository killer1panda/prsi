## 2023-10-27 - Accessible Scrollable Regions in Next.js UI Cards
**Learning:** By default, standard CardContent containers with overflow-y-auto are inaccessible via keyboard, breaking the a11y experience for output streams.
**Action:** When adding overflow-y-auto to containers displaying critical data streams (like Inference Output Stream), always add tabIndex={0}, role="region", aria-label="...", and focus-visible:ring-2 to make them properly keyboard accessible and readable by screen readers.
