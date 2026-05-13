select d_year,
       sum(case when lo_quantity < 25 then lo_revenue else 0 end) as low_qty_revenue,
       sum(case lo_discount when 1 then lo_revenue else 0 end) as discount_1_revenue
from lineorder, ddate
where lo_orderdate = d_datekey
group by d_year
order by low_qty_revenue desc
limit 5;
