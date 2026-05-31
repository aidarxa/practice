select d_year,c_nation,sum(lo_revenue-lo_supplycost) as profit
from lineorder_flat
where c_region = 1
and s_region = 1
and (p_mfgr = 0 or p_mfgr = 1)
group by d_year,c_nation;
