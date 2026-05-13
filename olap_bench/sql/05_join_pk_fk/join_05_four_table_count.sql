select count(*)
from lineorder, part, supplier, ddate
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 1
and s_region = 1;
