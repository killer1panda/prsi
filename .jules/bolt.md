## 2024-06-03 - Avoid `iterrows` in Pandas transformations
**Learning:** `pd.DataFrame.iterrows()` can introduce massive overhead for row-by-row traversal because it constructs a Pandas Series object on every iteration.
**Action:** Extract the column to a direct list or iter over `.values` / standard series iteration. Also avoid repeatedly calling `.str.lower()` inside loops over large arrays for string comparisons, and supply `regex=False` to `.str.contains` when possible.
