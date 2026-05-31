select lo_revenue - lo_supplycost, c_nation
from lineorder_flat
where c_region = 1;
