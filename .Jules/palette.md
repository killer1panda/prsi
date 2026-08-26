## 2024-08-26 - Accessibility of Dynamically Loaded Analysis Results
**Learning:** When async analysis results are loaded into the DOM (like cancellation risk and sentiment scores in the Doom Index analyzer), screen readers miss the update completely if the container isn't marked as a live region.
**Action:** Always add `aria-live="polite"` to result containers that are populated via fetch/async operations so screen readers can announce the changes.
