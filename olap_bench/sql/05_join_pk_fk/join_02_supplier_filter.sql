select count(*)
from lineorder, supplier
where lo_suppkey = s_suppkey
and s_region = 1;
