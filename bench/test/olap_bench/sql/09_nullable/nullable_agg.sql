-- Synthetic nullable semantics test. Requires a test dataset where
-- lo_revenue has NULLs at known positions.
SELECT count(*), count(lo_revenue), sum(lo_revenue), avg(lo_revenue)
FROM lineorder;
