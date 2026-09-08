## 2024-03-24 - Make Scrollable Containers Keyboard Accessible
**Learning:** Custom scrollable containers (like those using `overflow-y-auto`) are not keyboard-accessible by default. Keyboard-only users cannot scroll the content unless they can focus the container itself.
**Action:** Always add `tabIndex={0}`, `role="region"`, and an `aria-label` to custom scrollable containers, along with `focus-visible` styling, to ensure they can be focused and announced properly by screen readers.
## 2024-03-25 - Avoid using text-rose-900 on dark backgrounds
**Learning:** The color `text-rose-900` has an extremely poor color contrast ratio (~1.5:1) against the dark theme background (`bg-zinc-950`), making the text essentially invisible.
**Action:** Use a lighter color like `text-rose-500/50` or `text-rose-400` when working with red accents on dark backgrounds to ensure adequate accessibility and readability. Always test color contrast in the active theme.
