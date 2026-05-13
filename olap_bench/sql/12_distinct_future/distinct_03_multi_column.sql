select distinct d_year, p_brand1
from lineorder, ddate, part
where lo_orderdate = d_datekey
and lo_partkey = p_partkey;
