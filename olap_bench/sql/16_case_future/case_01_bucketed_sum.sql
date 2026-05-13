select case when lo_quantity < 25 then 0 else 1 end, sum(lo_revenue)
from lineorder
group by case when lo_quantity < 25 then 0 else 1 end;
