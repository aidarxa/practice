select count(*)
from lineorder_flat
where lo_discount = 1 or lo_discount = 2;
