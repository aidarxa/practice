-- Expected on standard SSB: 0.
-- Expected on synthetic nullable data: number of NULL lo_revenue rows.
SELECT count(*)
FROM lineorder
WHERE lo_revenue IS NULL;
