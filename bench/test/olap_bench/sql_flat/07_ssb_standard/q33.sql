select c_city,s_city,d_year,sum(lo_revenue) as revenue
from lineorder_flat
where (c_city = 231 or c_city = 235)
and (s_city = 231 or s_city = 235)
and d_year >=1992 and d_year <= 1997
group by c_city,s_city,d_year;
