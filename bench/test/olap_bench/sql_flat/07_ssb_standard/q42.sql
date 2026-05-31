select d_year,s_nation,p_category,sum(lo_revenue-lo_supplycost) as profit
from lineorder_flat
where c_region = 1
and s_region = 1
and (d_year = 1997 or d_year = 1998)
and (p_mfgr = 0 or p_mfgr = 1)
group by d_year,s_nation, p_category;
