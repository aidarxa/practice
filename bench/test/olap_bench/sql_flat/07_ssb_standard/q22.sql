select sum(lo_revenue),d_year,p_brand1
from lineorder_flat
where p_brand1 >= 260
and p_brand1 <= 267
and s_region = 2
group by d_year,p_brand1;
