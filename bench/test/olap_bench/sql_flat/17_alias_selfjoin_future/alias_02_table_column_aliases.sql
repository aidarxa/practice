select d_year as order_year, sum(lo_revenue) as revenue
from lineorder_flat
group by d_year
having revenue > 1000000000000
order by revenue desc
limit 5;
