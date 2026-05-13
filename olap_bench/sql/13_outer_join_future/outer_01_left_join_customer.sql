select lo_orderkey, c_nation
from lineorder left join customer on lo_custkey = c_custkey;
