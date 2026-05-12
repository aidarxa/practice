SELECT d_year, sum(lo_revenue)
FROM lineorder, customer, ddate
WHERE lo_custkey = c_custkey
  AND lo_orderdate = d_datekey
  AND (
    (c_mktsegment = 1 AND d_year = 1998)
    OR
    (c_mktsegment = 2 AND d_year = 1997)
  )
GROUP BY d_year;