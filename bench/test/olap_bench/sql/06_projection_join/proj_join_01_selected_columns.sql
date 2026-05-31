select lo_orderkey, c_nation, c_region
from lineorder, customer
where lo_custkey = c_custkey
and c_region = 1;
