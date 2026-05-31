SELECT d_year, sum(lo_revenue)
from lineorder_flat
where (
    (c_mktsegment = 1 AND d_year = 1998)
    OR
    (c_mktsegment = 2 AND d_year = 1997)
  )
GROUP BY d_year;
