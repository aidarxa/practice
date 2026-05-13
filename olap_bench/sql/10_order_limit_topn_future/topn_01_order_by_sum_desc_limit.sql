select d_year, sum(lo_revenue)
from lineorder, ddate
where lo_orderdate = d_datekey
group by d_year
order by sum(lo_revenue) desc
limit 5;
