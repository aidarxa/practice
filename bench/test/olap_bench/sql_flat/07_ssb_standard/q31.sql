select c_nation,s_nation,d_year,sum(lo_revenue) as revenue
from lineorder_flat
where c_region = 2
and s_region = 2
and d_year >= 1992 and d_year <= 1997
group by c_nation,s_nation,d_year;
