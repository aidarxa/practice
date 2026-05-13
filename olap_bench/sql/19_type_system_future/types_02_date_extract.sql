select extract(year from lo_orderdate), count(*) from lineorder group by extract(year from lo_orderdate);
