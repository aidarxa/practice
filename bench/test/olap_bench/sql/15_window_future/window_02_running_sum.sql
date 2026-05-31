select d_year, lo_orderkey, sum(lo_revenue) over (partition by d_year order by lo_orderkey)
from lineorder, ddate
where lo_orderdate = d_datekey;
