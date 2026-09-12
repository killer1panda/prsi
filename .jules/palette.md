## 2024-09-12 - Added Accessible Name to LiveFeedPanel
**Learning:** Custom scrollable containers ('overflow-y-auto') without focusable elements create 'keyboard traps' or bypassable regions. They need 'tabIndex={0}', 'role="region"', and 'aria-label' to be properly exposed to assistive technologies and reachable via keyboard navigation.
**Action:** Add these attributes and 'focus-visible' styling to all custom scrollable sections.
