select d_year, sum(lo_revenue)
from lineorder_flat
group by d_year
having sum(lo_revenue) > 1000000000000;
