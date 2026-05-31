SELECT c_nation, sum(lo_revenue)
from lineorder_flat
where c_nation = s_nation
  AND c_region = 1
GROUP BY c_nation;
