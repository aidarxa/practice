select lo_revenue - lo_supplycost, c_nation
from lineorder, customer
where lo_custkey = c_custkey
and c_region = 1;
