-- Expected on standard SSB: all rows are 0 because lo_revenue is non-nullable.
-- Expected on synthetic nullable data: 1 exactly for rows with NULL lo_revenue.
SELECT lo_revenue IS NULL
FROM lineorder;
