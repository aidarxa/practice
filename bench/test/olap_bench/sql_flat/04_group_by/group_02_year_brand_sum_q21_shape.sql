select sum(lo_revenue), d_year, p_brand1
from lineorder_flat
where p_category = 1
and s_region = 1
group by d_year, p_brand1;
