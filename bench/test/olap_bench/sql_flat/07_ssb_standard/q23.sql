select sum(lo_revenue),d_year,p_brand1
from lineorder_flat
where p_brand1 = 260
and s_region = 3
group by d_year,p_brand1;
