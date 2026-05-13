select d.d_year as order_year, sum(lo.lo_revenue) as revenue
from lineorder lo, ddate d
where lo.lo_orderdate = d.d_datekey
group by d.d_year
having revenue > 1000000000000
order by revenue desc
limit 5;
