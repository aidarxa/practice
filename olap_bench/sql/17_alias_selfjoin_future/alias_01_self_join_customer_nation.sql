select c1.c_nation, count(*)
from customer c1, customer c2
where c1.c_nation = c2.c_nation
group by c1.c_nation;
