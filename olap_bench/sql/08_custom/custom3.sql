SELECT c_nation, sum(lo_revenue)
FROM lineorder, customer, supplier
WHERE lo_custkey = c_custkey
  AND lo_suppkey = s_suppkey
  AND c_nation = s_nation
  AND c_region = 1
GROUP BY c_nation;