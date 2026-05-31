select lo_orderkey, lo_revenue
from lineorder_flat
order by lo_revenue desc limit 100;
