select case when lo_quantity < 10 then 0 when lo_quantity < 25 then 1 else 2 end as qty_bucket,
       sum(lo_revenue) as revenue
from lineorder
group by case when lo_quantity < 10 then 0 when lo_quantity < 25 then 1 else 2 end
order by revenue desc;
