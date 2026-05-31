select lo_orderkey, c_nation, s_nation, lo_revenue
from lineorder, customer, supplier
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and c_nation = s_nation;
