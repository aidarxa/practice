-- q1
select count(*), count(val), sum(val), avg(val) from nullable_numbers;

-- q2
select grp, count(*), count(val), sum(val)
from nullable_numbers
group by grp
order by grp;

-- q3
select count(*) from nullable_numbers where val is null;

-- q4
select count(*) from nullable_numbers where flag = 1;

-- q5
select grp, sum(case when val is null then 1 else 0 end) as null_count
from nullable_numbers
group by grp
order by grp;
