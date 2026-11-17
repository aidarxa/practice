select c_nation, s_nation, sum(lo_revenue)
from lineorder, customer, supplier
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and c_region = 'AMERICA'
and s_region = 'AMERICA'
group by c_nation, s_nation;
