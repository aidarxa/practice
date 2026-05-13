-- Requires a synthetic non-unique dimension table to exercise MHT row expansion.
-- The expected semantics are SQL INNER JOIN semantics: one probe row expands to
-- every matching build row.
select *
from lineorder, customer
where lo_custkey = c_custkey;
