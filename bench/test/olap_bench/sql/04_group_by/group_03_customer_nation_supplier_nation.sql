select c_nation, s_nation, sum(lo_revenue)
from lineorder, customer, supplier
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and c_region = 1
and s_region = 1
group by c_nation, s_nation;
