select count(*)
from lineorder, part
where lo_partkey = p_partkey
and p_category = 1;
