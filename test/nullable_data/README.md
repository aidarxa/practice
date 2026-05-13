# Nullable synthetic test data convention

The runtime nullable loader accepts validity or null bitmaps next to the column files:

- `<physical-column-file>.valid64`: uint64_t bitset, 1 = valid, 0 = NULL
- `<physical-column-file>.null64`: uint64_t bitset, 1 = NULL, 0 = valid
- `<physical-column-file>.valid8`: one byte per row, nonzero = valid
- `<physical-column-file>.null8`: one byte per row, nonzero = NULL

For `lo_revenue`, the physical column file is `LINEORDER12` according to `lookup()`.
A synthetic nullable SF10 test can therefore add one of:

- `data/10/LINEORDER12.valid64`
- `data/10/LINEORDER12.null64`
- `data/10/LINEORDER12.valid8`
- `data/10/LINEORDER12.null8`

The SQL smoke tests in `test/queries/sql/nullable/` verify `COUNT(expr)`,
`SUM`, `AVG`, `IS NULL`, and `WHERE ... IS NULL` against those bitmaps.
