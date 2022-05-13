---
layout: single
title: "SQL 복습"
date: 2022-04-19 18:22:01
lastmod : 2022-05-13 12:14:46
categories: sql
tag: [sql]
toc: true
toc_sticky: true
---

# 1
`WHERE` 절에서 `= NULL` 이 아니라 `IS NULL`로 찾아야 한다. 반대는 `IS NOT NULL`

# 2
```sql
SELECT count(DISTINCT ID)
from teaches;
```
`teaches` 테이블에서 `ID`의 종류의 개수를 센다

```sql
SELECT count(ID)
from teaches;
```
`teaches` 테이블에서 `ID`의 개수를 센다. (중복 포함)

# 3

```sql
SELECT dept_name, avg(salary), ID
FROM instructor
GROUP BY dept_name;
```
여기서 `ID`는 오류를 발생시키거나 아무거나 출력한다.

# 4
* `having` : 집계함수를 취한 다음에 적용
* `where` : 집계함수 이전에 적용, `WHERE` 뒤에 `GROUP BY`를 쓰면 조건식을 통과한 Tuple만 집계함수에 포함된다.

# 5
하나는 오름 차순 하나는 내림차순 할 때
```sql
...
ORDER BY A, B desc
```

# 6
상위 N개만 출력할 때
```sql
...
LIMIT N
```

# 7
`NAME` Column에 NULL 이 하나 있고 전체 row는 4라 가정

```sql
SELECT COUNT(*)
...
```

출력 : 4

```sql
SELECT COUNT(NAME)
```
출력 : 3

`NAME`에서 `NULL`인 개수를 세려면
```sql
SELECT COUNT(NAME)
...
WHERE NAME IS NULL
```
이 아니라
```sql
SELECT COUNT(*)
...
WHERE NAME IS NULL
```
로 해야한다.

# 8
`GROUP BY`와 집계함수를 같이 쓸 경우 집계함수의 매개변수로 `GROUP BY`에 들어간 column을 쓰도록 하자

# 9
`HAVING` VS `WHERE`

predicates in the having clause are applied after the formation of groups whereas predicates in the where clause are applied before forming
groups
## Having
* 그룹 전체 즉, 그룹을 나타내는 결과 집합의 행에만 적용된다
* SQL select문이 집계 값이 지정된 조건을 충족하는 행만 반환하도록 지정하는 SQL절이다.
* 그룹을 필터링하는 데 사용된다.
* 집계 함수는 having 절과 함께 사용할 수 있다.
* Group By 절 뒤에 사용합니다.
## WHERE
* 개별 행에 적용이 된다.
* 단일 테이블에서 데이터를 가져 오거나 여러 테이블과 결합하여 조건을 지정하는데 사용되는 SQL절이다.
* 행을 필터링 하는데 사용된다.
* where절을 have절에 포함된 하위 쿼리에 있지않으면 집계함수와 함께 사용할 수 없다.
* Group By 절 앞에 사용합니다.

> https://velog.io/@ljs7463/SQL-having-%EA%B3%BC-where-%EC%B0%A8%EC%9D%B4

# 10
`A` column이 `datetime` 형식으로 되어 있다면 `HOUR(A)`을 할 경우 `A`의 시간만 추출한다.

# 11
`NULL` 대체하기

```sql
CASE WHEN A IS NULL 
    THEN 'REPLACE WORD' 
    ELSE A
END
```
```sql
COALESCE(Column, replace element)
```

# 12
## NATURAL JOIN
Natural join matches tuples with the same values for all common attributes, and retains only one copy of each common column.

```sql
SELECT Column1, Column2...
FROM table_name1 NATURAL JOIN table_name2
[NATURAL JOIN table_....]
WHERE Condition;
```

`NATURAL JOIN`은 `ON` 이 없다.

# 13
```sql
...
LLL LEFT OUTER JOIN RRR
WHERE RRR.ID IS NULL
```
```sql
...
WHERE ID NOT IN (SELECT ID FROM RRR)
```

# 14
조건을 이용한 Column만들기

```sql
SELECT ..., CASE
                WHEN CONDITION1 THEN OUTPUT1
                WHEN CONDITION2 THEN OUTPUT2
                ...
                WHEN CONDITIONn THEN OUTPUTn
                ELSE OUTPUTk
            END AS COLUMN_NAME
...
```

# 15
`DATETIME`을 원하는 형식으로 출력하기

```sql
SELECT *, date_format(Column_name1, '%Y-%m-%d')
```

# 16

https://www.hackerrank.com/challenges/earnings-of-employees/problem

```sql
SELECT salary * months AS earnings, COUNT(*)
FROM Employee
GROUP BY earnings
ORDER BY earnings DESC
LIMIT 1;
```

# 17
https://www.hackerrank.com/challenges/the-blunder/problem?isFullScreen=true

*[특정 문자를 변경해서 표시]*
```
SELECT REPLACE(field_name, TARGET, NEW)
FROM table_name;
```
*[특정 문자를 변경해서 업데이트]*
```
UPDATE 테이블명 SET 
필드명 = REPLACE(필드명, TARGET, NEW);
WHERE 조건문
```
*[ex]*
```sql
/*  -- 테이블 submachtbl 에서 */
UPDATE `submachtbl` SET
/* machID 필드에서 'mc1'를 'mc2'로 변경한다. */
machID = REPLACE(machID, 'mc1', 'mc2')
/* 단, machID는 'mc1'이고 machName은 'newMc'에 한해서 */
WHERE machID='mc1' AND machName='newMc';
```

```sql
SELECT CEIL(AVG(salary) - AVG(replace(salary,0,'')))
FROM employees
```

