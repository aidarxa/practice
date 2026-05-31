select c_city,s_city,d_year,sum(lo_revenue) as revenue
from lineorder_flat
where c_nation = 24
and s_nation = 24
and d_year >=1992 and d_year <= 1997
group by c_city,s_city,d_year;
