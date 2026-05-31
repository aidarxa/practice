-- Expected on standard SSB: 0.
-- Expected on synthetic nullable data: number of NULL lo_revenue rows.
SELECT count(*)
from lineorder_flat
where lo_revenue IS NULL;
