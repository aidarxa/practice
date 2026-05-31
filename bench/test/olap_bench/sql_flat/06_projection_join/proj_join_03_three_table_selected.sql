select lo_orderkey, c_nation, s_nation, lo_revenue
from lineorder_flat
where c_nation = s_nation;
