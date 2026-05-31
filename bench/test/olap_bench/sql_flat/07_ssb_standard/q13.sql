select sum(lo_extendedprice * lo_discount) as revenue
from lineorder_flat
where lo_orderdate >= 19940204
and lo_orderdate <= 19940210
and lo_discount>=5
and lo_discount<=7
and lo_quantity>=26
and lo_quantity<=35;
