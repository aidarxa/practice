SELECT count(lo_orderkey)
from lineorder_flat
where lo_quantity >= p_size - 5
  AND lo_quantity <= p_size + 5;
