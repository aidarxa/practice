with yearly as (
  select d_year, sum(lo_revenue) as total_revenue
  from lineorder, ddate
  where lo_orderdate = d_datekey
  group by d_year
)
select d_year, total_revenue from yearly;
