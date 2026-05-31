select count(*)
from lineorder_flat
where not (lo_revenue is null);
