select c_city, count(*)
from customer, supplier
where c_city = s_city
group by c_city;
