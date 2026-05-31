-- Smoke tests for SQL three-valued logic over nullable columns.
select count(*) from lineorder where not (lo_revenue is null);
select lo_revenue is null or lo_revenue > 0 from lineorder;
select lo_revenue is null and lo_revenue > 0 from lineorder;
