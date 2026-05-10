SELECT count(lo_orderkey)
FROM lineorder, part
WHERE lo_partkey = p_partkey
  AND lo_quantity >= p_size - 5 
  AND lo_quantity <= p_size + 5;