select lo_quantity + lo_discount, count(*) from lineorder group by lo_quantity + lo_discount;
