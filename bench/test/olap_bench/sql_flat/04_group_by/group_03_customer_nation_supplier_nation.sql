select c_nation, s_nation, sum(lo_revenue)
from lineorder_flat
where c_region = 1
and s_region = 1
group by c_nation, s_nation;
