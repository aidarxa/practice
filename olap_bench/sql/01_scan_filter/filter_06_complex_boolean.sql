select count(*) from lineorder where (lo_quantity >= 10 and lo_quantity <= 20) or (lo_discount >= 2 and lo_discount <= 4);
