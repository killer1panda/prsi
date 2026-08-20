## 2024-08-20 - Redundant String Manipulation in Pandas
**Learning:** Found a performance bottleneck in `src/features/engineering.py` where a `.fillna('').str.lower()` operation on a text column was being executed inside a loop for each keyword in a list, resulting in N passes over the column.
**Action:** Always hoist invariant Series string manipulations outside of loops in pandas. Also, use `regex=False` in `str.contains()` when doing simple substring matches, as it significantly bypasses regex engine overhead.
