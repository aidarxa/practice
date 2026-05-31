select count(*)
from lineorder_flat
where lo_quantity * 2 < lo_discount * 10;
