select count(*)
from lineorder, ddate
where lo_orderdate = d_datekey
and d_year = 1997;
