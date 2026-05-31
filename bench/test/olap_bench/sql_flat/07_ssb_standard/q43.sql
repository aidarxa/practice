select d_year,s_city,p_brand1,sum(lo_revenue-lo_supplycost) as profit
from lineorder_flat
where c_region = 1
and s_nation = 24
and (d_year = 1997 or d_year = 1998)
and p_category = 3
group by d_year,s_city,p_brand1;
