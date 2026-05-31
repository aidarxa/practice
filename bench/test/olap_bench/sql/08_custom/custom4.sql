SELECT c_region, sum(lo_extendedprice * lo_discount) as profit
FROM lineorder, customer
WHERE lo_custkey = c_custkey
  AND (
    (lo_quantity * 2 < lo_discount * 10)
    OR
    (lo_tax + lo_discount >= 5 AND lo_tax + lo_discount <= 10)
    OR
    (c_region = 2 AND lo_revenue > 1000000)
  )
GROUP BY c_region;