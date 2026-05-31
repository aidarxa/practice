select count(*)
from lineorder_flat
where lo_quantity >= 25 and lo_quantity <= 35;
