SELECT d_year, sum(lo_revenue)
FROM lineorder, ddate
WHERE 
  (lo_orderdate = d_datekey OR lo_commitdate = d_datekey)
  AND d_year >= 1992 AND d_year <= 1997
GROUP BY d_year;