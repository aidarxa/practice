select d_year, sum(lo_revenue)
from lineorder_flat
group by d_year
order by sum(lo_revenue) desc
limit 5;
