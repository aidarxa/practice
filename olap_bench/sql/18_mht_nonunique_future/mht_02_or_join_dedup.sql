select d_year, count(*)
from lineorder, ddate
where lo_orderdate = d_datekey or lo_commitdate = d_datekey
group by d_year;
