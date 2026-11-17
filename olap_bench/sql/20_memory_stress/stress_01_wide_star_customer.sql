select *
from lineorder, customer
where lo_custkey = c_custkey
and c_region = 'AMERICA';
