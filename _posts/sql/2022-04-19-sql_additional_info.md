---
layout: single
title: "SQL 복습"
date: 2022-04-19 18:22:01
lastmod : 2022-04-19 18:22:04
categories: sql
tag: [sql]
toc: true
toc_sticky: true
---

# 2022-04-19 18:28:21
`WHERE` 절에서 `= NULL` 이 아니라 `IS NULL`로 찾아야 한다. 반대는 `IS NOT NULL`

# 2022-04-19 18:28:34
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

# 2022-04-19 18:30:40

```sql
SELECT dept_name, avg(salary), ID
FROM instructor
GROUP BY dept_name;
```
여기서 `ID`는 오류를 발생시키거나 아무거나 출력한다.

# 2022-04-19 18:34:52
* `having` : 집계함수를 취한 다음에 적용
* `where` : 집계함수 이전에 적용, `WHERE` 뒤에 `GROUP BY`를 쓰면 조건식을 통과한 Tuple만 집계함수에 포함된다.
