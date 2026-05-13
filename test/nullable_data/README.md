# Nullable mini dataset

This directory contains a deterministic nullable dataset used as a correctness oracle for the nullable SQL subset.

Null encoding convention for the engine remains:

- `<physical-column-file>.valid64`: uint64_t bitset, 1 = valid, 0 = NULL
- `<physical-column-file>.null64`: uint64_t bitset, 1 = NULL, 0 = valid
- `<physical-column-file>.valid8`: one byte per row, nonzero = valid
- `<physical-column-file>.null8`: one byte per row, nonzero = NULL

The mini dataset is intentionally stored as CSV plus expected query outputs so it can be loaded into DuckDB/ClickHouse or converted into the engine's physical column files.

Files:

- `schema.sql`: logical schema for the dataset.
- `nullable_numbers.csv`: 12-row input table; empty CSV fields represent SQL NULL.
- `queries.sql`: regression queries.
- `expected.tsv`: normalized expected output for each query.
