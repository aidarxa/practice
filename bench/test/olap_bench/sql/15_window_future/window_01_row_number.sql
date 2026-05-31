select d_year, lo_revenue, row_number() over (partition by d_year order by lo_revenue desc)
from lineorder, ddate
where lo_orderdate = d_datekey;
