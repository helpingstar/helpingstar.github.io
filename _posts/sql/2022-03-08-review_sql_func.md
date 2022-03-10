---
layout: single
title: "SQL 명령어 복습"
date: 2022-03-08 13:41:50
lastmod : 2022-03-08 13:41:48
categories: sql
tag: [sql]
toc: true
toc_sticky: true
---

# Login / LogOut
```sql
mysql -u root -p
```
루트 계정을 사용하여 로그인

```sql
mysql -u root -p sakila;
```
`mysql` 명령줄 도구를 호출할 때마다 사용자 이름과 데이터 베이스를 모두 지정할 수 있다. 이러면 도구를 시작할 때마다 `use sakila;`를 입력할 필요가 없다.

`quit;` 또는 `exit;`라고 입력하면 유닉스 또는 윈도우 명령 셸로 되돌아간다.

# `mysql` 명령줄 도구
```sql
mysql> SELECT now();
+---------------------+
| now()               |
+---------------------+
| 2022-03-08 13:51:26 |
+---------------------+
1 row in set (0.00 sec)
```
`mysql` 명령줄 도구는 `+`, `-` 및 `|` 문자로 구분된 사각형 안에 쿼리 결과를 보여줍니다.

결과를 보여준 뒤 `SQL`문 실행에 걸린 시간과 함께 리턴된 행의 수를 보여준다.

# `CREATE`
```sql
CREATE TABLE [Table Name]
(
    field_name field_type,
    ...
);
```

```sql
mysql> CREATE TABLE person
    -> (person_id SMALLINT UNSIGNED,
    -> fname VARCHAR(20),
    -> lname VARCHAR(20),
    -> eye_color ENUM('BR','BL','GR'),
    -> birth_date DATE,
    -> street VARCHAR(30),
    -> city VARCHAR(20),
    -> state VARCHAR(20),
    -> country VARCHAR(20),
    -> postal_code VARCHAR(20),
    -> CONSTRAINT pk_person PRIMARY KEY (person_id)
    -> );
Query OK, 0 rows affected (0.08 sec)
```
SQL문을 생성하여 데이터베이스에 테이블을 만든다.

## `Constraint`

## Primary Key Constraint
```sql
CREATE TABLE [Table Name]
(
    field_name field_type,
    ...
    CONSTRAINT [constraint_name] PRIMARY KEY field_name
)
```
```sql
mysql> CREATE TABLE person
    -> ...
    -> CONSTRAINT pk_person PRIMARY KEY (person_id)
    -> );
Query OK, 0 rows affected (0.08 sec)
```
테이블 정의 시 기본 키로 사용할 열을 데이터베이스 서버에 알려줘야 하므로 테이블에 제약조건을 만들어 이 작업을 수행한다. 현재 추가한 제약 조건은 **기본 키 제약조건**(Primary Key Constraint) 이다. `person_id` 열에 생성되며 `pk_person`이라는 이름이 지정된다.

## Foreign Key Constraint

```sql
CREATE TABLE [Table_name]
(
    [Field_name] [Field_type],
    ...,
    [CONSTRAINT [Constraint_name]] FOREIGN KEY ([Field_name])
    REFERENCES [Table_name] ([Field_name])
)
```

```sql
mysql> CREATE TABLE favorite_food
    -> (person_id SMALLINT UNSIGNED,
    -> food VARCHAR(20),
    -> CONSTRAINT pk_favorite_food PRIMARY KEY (person_id, food),
    -> CONSTRAINT fk_fav_food_person_id FOREIGN KEY (person_id)
    -> REFERENCES person (person_id)
    -> );
Query OK, 0 rows affected (0.09 sec)
```
`favorite_food` 테이블에서 `person_id` 열의 값에 `person` 테이블에 있는 값만 포함되도록 제한된다.

## Check Constraint
특정 열에 허용되는 값을 제한한다.
```sql
eye_color CHAR(2) CHECK (eye_color IN ('BR', 'BL', 'GR')),
```

특정 열에 대해 허용 가능한 값을 표시한다. (ex. 'BR', 'BL', ...)

```sql
eye_color ENUM('BR','BL','GR'),
```
MySQL은 체크 제약 조건을 자료형 정의에 병합하는 `enum`이라는 자료형을 제공한다.

# `describe(desc)`
테이블의 정의를 볼 수 있다.
```sql
mysql> desc person;
+-------------+----------------------+------+-----+---------+-------+
| Field       | Type                 | Null | Key | Default | Extra |
+-------------+----------------------+------+-----+---------+-------+
| person_id   | smallint unsigned    | NO   | PRI | NULL    |       |
| fname       | varchar(20)          | YES  |     | NULL    |       |
| lname       | varchar(20)          | YES  |     | NULL    |       |
| eye_color   | enum('BR','BL','GR') | YES  |     | NULL    |       |
| birth_date  | date                 | YES  |     | NULL    |       |
| street      | varchar(30)          | YES  |     | NULL    |       |
| city        | varchar(20)          | YES  |     | NULL    |       |
| state       | varchar(20)          | YES  |     | NULL    |       |
| country     | varchar(20)          | YES  |     | NULL    |       |
| postal_code | varchar(20)          | YES  |     | NULL    |       |
+-------------+----------------------+------+-----+---------+-------+
10 rows in set (0.01 sec)
```
* `Null` : 데이터를 테이블에 삽입할 때 특정 열의 데이터를 생략할 수 있는지, `not null` 키워드를 추가하여 `null`의 허용 여부를 지정할 수 있다.
* `Key` : 기본 키나 외래 키에 포함되는지 여부를 나타냄 (`person_id`)
* `Default` : 테이블에 데이터를 삽입할 때 특정 열을 생략할 경우 해당 열을 * 기본값으로 채울 지
* `Extra` : 열에 적용될 수 있는 기타 정보

# `ALTER TABLE`

## `MODIFY`
```sql
ALTER TABLE person MODIFY person_id SMALLINT UNSIGNED AUTO_INCEREMENT;
```

```sql
mysql> desc person;
+-------------+------------------ +------+-----+---------+-----------------+
| Field       | Type              | Null | Key | Default | Extra           |
+-------------+------------------ +------+-----+---------+-----------------+
| person_id   | smallint unsigned | NO   | PRI | NULL    | auto_incerement |
+-------------+-------------------+------+-----+---------+-----------------+
...
```
`person` 테이블의 `person_id` 열을 재정의한다. 

`person` 테이블에 데이터를 삽입할 때 `person_id` 열에 `null` 값을 제공하기만 하면 MySQL은 사용 가능한 다음 숫자로 열을 채운다. (1-based)



# `INSERT`
```sql
INSERT INTO [table_name]
([field_name1], [field_name2], [field_name3], ...)
VALUES ([field_name1], [field_name2], [field_name3], ...)
```

```sql
mysql> INSERT INTO person
    -> (person_id, fname, lname, eye_color, birth_date)
    -> VALUES (null, 'William', 'Turner', 'BR', '1972-05-27');
Query OK, 1 row affected (0.01 sec)
```
* 값을 제공하지 않은 열이 있지만, 해당 열들은 `null` 이 허용되므로 문제 없다.
* `birth_date` 열에 제공된 값은 문자열이다 데이터 타입인 `date`의 필수 형식과 일치할 경우 MySQL은 문자열을 날짜로 변환한다.
* 열 이름과 값은 각각 그 개수와 자료형이 일치해야 한다. 7개의 열에 이름을 지정하고 6개의 값만 제공하거나, 해당 열에 적합한 자료형으로 변환할 수 없는 값을 제공하면 문제가 발생한다.

# 쿼리 절
`select` 문은 여러 개의 구성요소 및 절로 구성된다

필수 항목은 `select`절 하나 뿐이지만, 보통은 사용 가능한 6개의 절 중 2개 또는 3개 이상의 절이 포함된다.

|Chase name|Purpose|
|-|-|
|`select`|쿼리 결과에 포함할 열을 결정한다.|
|`from`|데이터를 검색할 테이블과, 테이블을 조인하는 방법을 식별한다.|
|`where`|불필요한 데이터를 걸러낸다.|
|`group by`|공통 열 값을 기준으로 행을 그룹화한다.|
|`having`|불필요한 그룹을 걸러낸다.|
|`order by`|하나 이상의 열을 기준으로 최종 결과의 행을 정렬한다.|

## `SELECT`
`select` 문의 첫 번째 절이지만 데이터베이스 서버가 판단하는 마지막 절 중 하나이다. 최종 결과셋에 포함할 항목을 결정하려면 최종 결과셋에 포함될 수 있는 모든 열을 먼저 알아야 하기 때문이다.

**설명 1**

```sql
SELECT [field_name]
FROM [table_name]
WHERE condition
ORDER BY [field_name]
```

```sql
mysql> SELECT person_id, fname, lname, birth_date
    -> FROM person
    (-> WHERE person_id = 1) or (-> WHERE lname = 'Turner';)
+-----------+---------+--------+------------+
| person_id | fname   | lname  | birth_date |
+-----------+---------+--------+------------+
|         1 | William | Turner | 1972-05-27 |
+-----------+---------+--------+------------+
1 row in set (0.00 sec)
```

```sql
mysql> SELECT food
    -> FROM favorite_food
    -> WHERE person_id = 1
    -> ORDER BY food;
+---------+
| food    |
+---------+
| cookies |
| nachos  |
| pizza   |
+---------+
3 rows in set (0.01 sec)
```

`order by` 절은 서버가 쿼리에서 반환한 데이터를 정렬하는 방법을 알려준다. `oder by` 절이 없으면 테이블의 데이터가 특정 순서로 검색된다는 보장이 없다.

**설명 2**
```sql
mysql> SELECT * FROM language;
+-------------+----------+---------------------+
| language_id | name     | last_update         |
+-------------+----------+---------------------+
|           1 | English  | 2006-02-15 05:02:19 |
|           2 | Italian  | 2006-02-15 05:02:19 |
...
6 rows in set (0.04 sec)
```
* `from` : `language`라는 단일 테이블을 나열한다

* `select` : `language` 테이블의 모든(별표 `*` 문자로 모든 열을 지정) 열이 결과에 포함되어야 함을 나타냄

* 종합 : `language` 테이블의 모든 열과 모든 행을 보여주세요

별표 `*` 문자로 모든 열을 지정하는 것 외에도 관심있는 열의 이름을 명시하여 지정할 수 있다.
```sql
mysql> SELECT language_id, name
    -> FROM language;
+-------------+----------+
| language_id | name     |
+-------------+----------+
|           1 | English  |
|           2 | Italian  |
...
6 rows in set (0.00 sec)
```
`select` 절 : 모든 열 중에 쿼리 결과에 포함한 열을 결정하는 역할

`select` 절에 포함할 수 있는 항목
* 숫자 또는 문자열과 같은 **리터럴**
* `transaction.amount * -1`과 같은 **표현식**
* `ROUND(transaction.amount, 2)`와 같은 **내장 함수** 호출
* **사용자 정의 함수** 호출

*[단일 쿼리에서 테이블 열, 리터럴, 표현식 및 내장 함수 호출]*

```sql
mysql> SELECT language_id,
    ->   'COMMON' language_usage,
    ->   language_id * 3.1415927 lang_pi_value,
    ->   upper(name) language_name
    -> FROM language;
+-------------+----------------+---------------+---------------+
| language_id | language_usage | lang_pi_value | language_name |
+-------------+----------------+---------------+---------------+
|           1 | COMMON         |     3.1415927 | ENGLISH       |
|           2 | COMMON         |     6.2831854 | ITALIAN       |
|           3 | COMMON         |     9.4247781 | JAPANESE      |
|           4 | COMMON         |    12.5663708 | MANDARIN      |
|           5 | COMMON         |    15.7079635 | FRENCH        |
|           6 | COMMON         |    18.8495562 | GERMAN        |
+-------------+----------------+---------------+---------------+
6 rows in set (0.01 sec)
```

내장 함수만 실행하거나 간단한 표현식을 사용하는 경우에는 다음과 같이 `from` 절을 완전히 건너뛸 수 있다.

```sql
mysql> SELECT version(),
    ->   user(),
    ->   database();
+-----------+----------------+------------+
| version() | user()         | database() |
+-----------+----------------+------------+
| 8.0.21    | root@localhost | sakila     |
+-----------+----------------+------------+
1 row in set (0.01 sec)
```

### **`AS`, 열의 별칭**
테이블에서 열에 새 레이블을 할당하고 싶거나 이름이 모호할 경우, 표현식 또는 내장 함수 호출로 생성된 결과의 해당 열에 레이블을 직접 지정할 수 있다. `select` 절의 각 요소 뒤에 **열 별칭**(column alias)을 추가하면 된다.

```sql
mysql> SELECT language_id,
    ->   'COMMON' language_usage,
    ->   language_id * 3.1415927 lang_pi_value,
    ->   upper(name) language_name
    -> FROM language;
+-------------+----------------+---------------+---------------+
| language_id | language_usage | lang_pi_value | language_name |
+-------------+----------------+---------------+---------------+
|           1 | COMMON         |     3.1415927 | ENGLISH       |
|           2 | COMMON         |     6.2831854 | ITALIAN       |
|           3 | COMMON         |     9.4247781 | JAPANESE      |
|           4 | COMMON         |    12.5663708 | MANDARIN      |
|           5 | COMMON         |    15.7079635 | FRENCH        |
|           6 | COMMON         |    18.8495562 | GERMAN        |
+-------------+----------------+---------------+---------------+
6 rows in set (0.01 sec)
```
2, 3, 4번째 열 뒤에 열 별칭인 `language_usage`, `lang_pi_value` 및 `language_name` 을 기입하여 열 별칭을 추가한다.

열 별칭을 사용하면 출력을 이해하기 쉽고, `mysql` 도구로 대화식 쿼리를 실행하는 대신 자바나 파이썬에서 쿼리를 실행할 때는 이렇게 작업하는 편이 더 수월하다.
`AS` 키워드를 사용하여 열 별칭을 더 두드러지게 할 수 있다.

```sql
mysql> SELECT language_id,
    ->   'COMMON' AS language_usage,
    ->   language_id * 3.1415927 AS lang_pi_value,
    ->   upper(name) AS language_name
    -> FROM language;
+-------------+----------------+---------------+---------------+
| language_id | language_usage | lang_pi_value | language_name |
+-------------+----------------+---------------+---------------+
|           1 | COMMON         |     3.1415927 | ENGLISH       |
|           2 | COMMON         |     6.2831854 | ITALIAN       |
...
6 rows in set (0.00 sec)
```
### **`distinct`, 중복 제거**
경우에 따라 쿼리가 중복된 데이터 행을 반환할 수 있다.

`distinct` 결과를 생성하려면 데이터를 정렬해야 하므로 결과셋의 용량이 클 떄는 시간이 오래 걸릴 수 있다. 
```sql
mysql> SELECT actor_id FROM film_actor ORDER BY actor_id;
+----------+
| actor_id |
+----------+
|        1 |
|        1 |
...
|      200 |
|      200 |
+----------+
5462 rows in set (0.02 sec)
```

만약 `actor_id`의 고유한 집합으로 보고싶다면 `select`키워드 바로 뒤에 `distinct` 키워드를 추가하여 확인할 수 있다.
```sql
mysql> SELECT DISTINCT actor_id FROM film_actor ORDER BY actor_id;
+----------+
| actor_id |
+----------+
|        1 |
|        2 |
...
|      199 |
|      200 |
+----------+
200 rows in set (0.00 sec)
```

서버가 중복 데이터를 제거하는 것을 원치 않거나 결과에 중복값이 없는 게 확실할 때는 `distinct`를 지정하는 대신 `all` 키워드를 지정할 수 있다. 그러나 `all` 키워드는 기본값이기에 대부분 포함하지 않는다.

## `FROM`
`from` 절은 쿼리에 **사용되는 테이블을 명시**할 뿐만 아니라, **테이블을 서로 연결**하는 수단도 함께 정의한다.

### **네 가지 유형의 테이블**
* 영구 테이블 : `create table` 문으로 생성
* 파생 테이블 : 하위 쿼리에서 반환하고 메모리에 보관된 행
* 임시 테이블 : 메모리에 저장된 휘발성 데이터
* 가상 테이블 : `create view` 문으로 생성

#### **파생 테이블**
서브쿼리는 다른 쿼리에 포함된 쿼리이다. 서브쿼리는 괄호로 묶여 있으며 `select` 문의 여러 부분에서 찾을 수 있다. 그러나 `from`절 내에서의 서브쿼리는 `from` 절에 명시된 다른 테이블과 상호작용할 수 있는 파생 테이블을 생성한다.

```sql
mysql> SELECT concat(cust.last_name, ', ', cust.first_name) full_name
    -> FROM
    ->   (SELECT first_name, last_name, email
    ->    FROM customer
    ->    WHERE first_name = 'JESSIE'
    ->   ) cust;
+---------------+
| full_name     |
+---------------+
| BANKS, JESSIE |
| MILAM, JESSIE |
+---------------+
2 rows in set (0.04 sec)
```
`customer` 테이블에 대한 서브쿼리는 3개의 열을 반환하고, 서브쿼리를 포함하는 쿼리는 이 3개의 열 중 2개를 참조한다. 서브쿼리는 별칭을 통해 참조되는데 이 경우에는 `cust`라고 지정했다. `cust`의 데이터는 쿼리 기간 동아 ㄴ메모리에 보관된 후 삭제된다.

#### **임시 테이블**
모든 관계형 데이터베이스는 휘발성의 임시 테이블을 정의할 수 있다. 이러한 테이블은 영구 테이블처럼 보이지만 임시 테이블에 삽입된 데이터는 어느 시점(보통 트랜잭션이 끝날 때 또는 데이터베이스 세션이 닫힐 때)에 사라진다.

*[성이 **J**로 시작하는 배우를 임시적으로 저장하는 예시]*
```sql
mysql> CREATE TEMPORARY TABLE actors_j
    -> (actor_id smallint(5),
    ->  first_name varchar(45),
    ->  last_name varchar(45)
    -> );
Query OK, 0 rows affected, 1 warning (0.00 sec)

mysql> INSERT INTO actors_j
    -> SELECT actor_id, first_name, last_name
    -> FROM actor
    -> WHERE last_name LIKE 'J%';
Query OK, 7 rows affected (0.02 sec)
Records: 7  Duplicates: 0  Warnings: 0

mysql> SELECT * FROM actors_j;
+----------+------------+-----------+
| actor_id | first_name | last_name |
+----------+------------+-----------+
|      119 | WARREN     | JACKMAN   |
|      131 | JANE       | JACKMAN   |
|        8 | MATTHEW    | JOHANSSON |
|       64 | RAY        | JOHANSSON |
|      146 | ALBERT     | JOHANSSON |
|       82 | WOODY      | JOLIE     |
|       43 | KIRK       | JOVOVICH  |
+----------+------------+-----------+
7 rows in set (0.00 sec)
```

위의 7개의 행은 일시적으로 메모리에 저장되며 세션이 종료되면 사라진다.

#### **가상 테이블(뷰)**

뷰는 데이터 딕셔너리에 저장된 쿼리이다. 마치 테이블처럼 도앚ㄱ하지만 뷰에 저장된 데이터가 존재하지는 않는다. 이 때문에 가상 테이블이라고 부른다. 뷰에 대해 쿼리를 실행하면 쿼리가 뷰 정의와 합쳐져 실행할 최종 쿼리를 만든다.

*[`employee` 테이블을 쿼리하여 4개의 열을 포함하는 뷰 정의]*
```sql
mysql> CREATE VIEW cust_vw AS
    -> SELECT customer_id, first_name, last_name, active
    -> FROM customer;
Query OK, 0 rows affected (0.02 sec)
```
뷰를 작성하더라도 데이터가 추가 생성되거나 저장되지는 않습니다. 서버는 이후 사용할 때 `select` 문 대신 뷰가 존재하므로 다음과 같이 뷰를 쿼리할 수 있습니다.

```sql
mysql> SELECT first_name, last_name
    -> FROM cust_vw
    -> WHERE active = 0;
+------------+-----------+
| first_name | last_name |
+------------+-----------+
| SANDRA     | MARTIN    |
| JUDITH     | COX       |
| SHEILA     | WELLS     |
| ERICA      | MATTHEWS  |
| HEIDI      | LARSON    |
| PENNY      | NEAL      |
| KENNETH    | GOODEN    |
| HARRY      | ARCE      |
| NATHAN     | RUNYON    |
| THEODORE   | CULP      |
| MAURICE    | CRAWLEY   |
| BEN        | EASTER    |
| CHRISTIAN  | JUNG      |
| JIMMIE     | EGGLESTON |
| TERRANCE   | ROUSH     |
+------------+-----------+
15 rows in set (0.02 sec)
```

사용자로부터 열을 숨기고 복잡한 데이터베이스 설계를 단순화하는 등 다양한 이유로 뷰가 만들어진다.


### 테이블 연결

단순 `from` 절 정의와 두 번째로 다른 점은 `from` 절에 둘 이상의 테이블이 있으면 그 테이블을 연결하는 데 필요한 조건도 포함해야 한다는 의무사항이다.

```sql
mysql> SELECT customer.first_name, customer.last_name,
    ->   time(rental.rental_date) rental_time
    -> FROM customer
    ->   INNER JOIN rental
    ->   ON customer.customer_id = rental.customer_id
    -> WHERE date(rental.rental_date) = '2005-06-14';
+------------+-----------+-------------+
| first_name | last_name | rental_time |
+------------+-----------+-------------+
| JEFFERY    | PINSON    | 22:53:33    |
| ELMER      | NOE       | 22:55:13    |
...
| CHARLES    | KOWALSKI  | 23:54:34    |
| JEANETTE   | GREENE    | 23:54:46    |
+------------+-----------+-------------+
16 rows in set (0.03 sec)
```

이전 쿼리는 `customer` 테이블의 열(`first_name`, `last_name`)과 `rental` 테이블의 열(`rental_date`) 데이터를 모두 표시하므로 두 테이블이 모두 `from` 절에 포함된다. 두 테이블을 연결(조인join) 하는 메커니즘은 `customer` 및 `rental` 테이블에 모두 저장된 `customer_id` 이다. 따라서 데이터베이스 서버는 `customer` 테이블의 `customer_id` 열 값을 사용하여 `rental` 테이블에서 모든 고객의 대여 내역을 찾도록 지시한다. 두 테이블의 조인 조건은 `from` 절의 하위 절에 있다. 이 경우 결합 조건은 `ON customer.customer_id = rental.customer_id` 이다. `where` 절은 조인의 일부가 아니며 `rental` 테이블에 16,000개가 넘는 행이 있으므로 결과를 최대한 좁혀 필터링하기 위해 포함된다.

### 테이블 별칭 정의
단일 쿼리에서 여러 테이블을 조인할 경우 `select`, `where`, `group by`, `have` 및 `order by` 절에서 열을 참조할 때 참조 테이블을 식별할 방법이 필요하다. `from` 절 외부에서 테이블을 참조할 때는 다음과 같은 두 가지 방법을 쓸 수 있다.
* `employee.emp_id`와 같이 전체 테이블 이름을 사용
* 각 테이블의 **별칭**을 할당하고 쿼리 전체에서 해당 별칭을 사용

[이전 쿼리](#테이블-연결)에서 `select` 및 `on` 절에 전체 테이블 이름을 사용했다. 다음은 테이블 별칭을 사용하는 동일한 쿼리이다.
```sql
mysql> SELECT c.first_name, c.last_name,
    ->   time(r.rental_date) rental_time
    -> FROM customer c
    ->   INNER JOIN rental r
    ->   ON c.customer_id = r.customer_id
    -> WHERE date(r.rental_date) = '2005-06-14';
+------------+-----------+-------------+
| first_name | last_name | rental_time |
+------------+-----------+-------------+
| JEFFERY    | PINSON    | 22:53:33    |
| ELMER      | NOE       | 22:55:13    |
...
| CHARLES    | KOWALSKI  | 23:54:34    |
| JEANETTE   | GREENE    | 23:54:46    |
+------------+-----------+-------------+
16 rows in set (0.00 sec)
```
`customer` 테이블에 별칭 `c`가 할당되고 `rental` 테이블에 별칭 `r`이 할당되었다. 이러한 별칭은 조인 조건을 정의하는 `on` 절과, 결과셋에 포함할 열을 지정하는 `select` 절에서 사용된다.

열 별칭과 마찬가지로 테이블 별칭에도 `as` 키워드를 사용할 수 있다.

```sql
SELECT c.first_name, c.last_name,
  time(r.rental_date) rental_time
FROM customer AS c
  INNER JOIN rental AS r
  ON c.customer_id = r.customer_id
WHERE date(r.rental_date) = '2005-06-14';
```

## `WHERE`
`where` 절은 결과셋에 출력되기를 원하지 않는 행을 필터링하는 메커니즘이다.

*[영화 대여에 관심이 있고 최소 일주일 동안 대여할 수 있는 G 등급의 영화]*

```sql
mysql> SELECT title, rating, rental_duration
    -> FROM film
    -> WHERE rating = 'G' AND rental_duration >= 7;
+-------------------------+--------+-----+
| title                   | rating | r_d |
+-------------------------+--------+-----+
| BLANKET BEVERLY         | G      |   7 |
| BORROWERS BEDAZZLED     | G      |   7 |
...
| WAKE JAWS               | G      |   7 |
| WAR NOTTING             | G      |   7 |
+-------------------------+--------+-----+
29 rows in set (0.00 sec)
```

각각의 조건은 `and`, `or` 또는 `not`과 같은 연산자로 구분된다.

`where` 절에 `and` 및 `or` 연산자를 모두 사용할 경우에는 (조건을 함께 그룹화 하기 위해서는) 괄호를 사용해야 한다.

*[G 등급이면서 7일 이상 대여할 수 있거나, PG-13 등급이면서 3일 이하로만 대여할 수 있는 영화]*

```sql
mysql> SELECT title, rating, rental_duration
    -> FROM film
    -> WHERE (rating = 'G' AND rental_duration >= 7)
    ->   OR (rating = 'PG-13' AND rental_duration < 4);
+-------------------------+--------+-----------------+
| title                   | rating | rental_duration |
+-------------------------+--------+-----------------+
| ALABAMA DEVIL           | PG-13  |               3 |
| BACKLASH UNDEFEATED     | PG-13  |               3 |
...
| WAR NOTTING             | G      |               7 |
| WORLD LEATHERNECKS      | PG-13  |               3 |
+-------------------------+--------+-----------------+
68 rows in set (0.00 sec)
```

## `GROUP BY`,  `HAVING`
* `GROUP BY` : 데이터를 열 값 별로 그룹화 한다.
* `HAVING` : `where` 절에서 원시 데이터를 필터링한다.

```sql
mysql> SELECT c.first_name, c.last_name, count(*)
    -> FROM customer c
    ->   INNER JOIN rental r
    ->   ON c.customer_id = r.customer_id
    -> GROUP BY c.first_name, c.last_name
    -> HAVING count(*) >= 40;
+------------+-----------+----------+
| first_name | last_name | count(*) |
+------------+-----------+----------+
| TAMMY      | SANDERS   |       41 |
| CLARA      | SHAW      |       42 |
| ELEANOR    | HUNT      |       46 |
| SUE        | PETERS    |       40 |
| MARCIA     | DEAN      |       42 |
| WESLEY     | BULL      |       40 |
| KARL       | SEAL      |       45 |
+------------+-----------+----------+
7 rows in set (0.04 sec)
```

## `ORDER BY`
`order by` 절은 원시 열 데이터 또는 열 데이터를 기반으로 표현식을 사용하여 결과셋을 정렬하는 메커니즘
```sql
SELECT column1, column2, ...
FROM table_name
ORDER BY column1, column2, ... ASC|DESC;
```

```sql
mysql> SELECT c.first_name, c.last_name,
    ->   time(r.rental_date) rental_date
    -> FROM customer c
    ->   INNER JOIN rental r
    ->   ON c.customer_id = r.customer_id
    -> WHERE date(r.rental_date) = '2005-06-14';
+------------+-----------+-------------+
| first_name | last_name | rental_date |
+------------+-----------+-------------+
| JEFFERY    | PINSON    | 22:53:33    |
| ELMER      | NOE       | 22:55:13    |
| MINNIE     | ROMERO    | 23:00:34    |
| MIRIAM     | MCKINNEY  | 23:07:08    |
| DANIEL     | CABRAL    | 23:09:38    |
| TERRANCE   | ROUSH     | 23:12:46    |
| JOYCE      | EDWARDS   | 23:16:26    |
| GWENDOLYN  | MAY       | 23:16:27    |
| CATHERINE  | CAMPBELL  | 23:17:03    |
| MATTHEW    | MAHAN     | 23:25:58    |
| HERMAN     | DEVORE    | 23:35:09    |
| AMBER      | DIXON     | 23:42:56    |
| TERRENCE   | GUNDERSON | 23:47:35    |
| SONIA      | GREGORY   | 23:50:11    |
| CHARLES    | KOWALSKI  | 23:54:34    |
| JEANETTE   | GREENE    | 23:54:46    |
+------------+-----------+-------------+
16 rows in set (0.01 sec)
```

`order by` 절에 `last_name` 열을 추가하여 성을 기준으로 알파벳순 정렬되도록 한다.

```sql
mysql> SELECT c.first_name, c.last_name,
    ->   time(r.rental_date) rental_date
    -> FROM customer c
    ->   INNER JOIN rental r
    ->   ON c.customer_id = r.customer_id
    -> WHERE date(r.rental_date) = '2005-06-14'
    -> ORDER BY c.last_name;
+------------+-----------+-------------+
| first_name | last_name | rental_date |
+------------+-----------+-------------+
| DANIEL     | CABRAL    | 23:09:38    |
| CATHERINE  | CAMPBELL  | 23:17:03    |
| HERMAN     | DEVORE    | 23:35:09    |
| AMBER      | DIXON     | 23:42:56    |
...
| ELMER      | NOE       | 22:55:13    |
| JEFFERY    | PINSON    | 22:53:33    |
| MINNIE     | ROMERO    | 23:00:34    |
| TERRANCE   | ROUSH     | 23:12:46    |
+------------+-----------+-------------+
16 rows in set (0.01 sec)
```
`order by` 절에서 `last_name` 열 뒤에 `first_name` 열을 추가하여 정렬 기준을 확장한다.

```sql
mysql> SELECT c.first_name, c.last_name,
    ->   time(r.rental_date) rental_date
    -> FROM customer c
    ->   INNER JOIN rental r
    ->   ON c.customer_id = r.customer_id
    -> WHERE date(r.rental_date) = '2005-06-14'
    -> ORDER BY c.last_name, c.first_name;
+------------+-----------+-------------+
| first_name | last_name | rental_date |
+------------+-----------+-------------+
| DANIEL     | CABRAL    | 23:09:38    |
| CATHERINE  | CAMPBELL  | 23:17:03    |
| HERMAN     | DEVORE    | 23:35:09    |
| AMBER      | DIXON     | 23:42:56    |
...
| ELMER      | NOE       | 22:55:13    |
| JEFFERY    | PINSON    | 22:53:33    |
| MINNIE     | ROMERO    | 23:00:34    |
| TERRANCE   | ROUSH     | 23:12:46    |
+------------+-----------+-------------+
16 rows in set (0.01 sec)
```

### `ASC` / `DESC`
`asc` 및 `desc` 키워드를 통해 오름차순 또는 내림차순을 지정할 수 있다. 기본값은 오름차순이다.

*[2005년 6월 14일에 영화를 대여한 모든 고객을 대여 시간의 내림차순으로 보여줌]*
```sql
mysql> SELECT c.first_name, c.last_name,
    ->   time(r.rental_date) rental_date
    -> FROM customer c
    ->   INNER JOIN rental r
    ->   ON c.customer_id = r.customer_id
    -> WHERE date(r.rental_date) = '2005-06-14'
    -> ORDER BY time(r.rental_date) desc;
+------------+-----------+-------------+
| first_name | last_name | rental_date |
+------------+-----------+-------------+
| JEANETTE   | GREENE    | 23:54:46    |
| CHARLES    | KOWALSKI  | 23:54:34    |
| SONIA      | GREGORY   | 23:50:11    |
| TERRENCE   | GUNDERSON | 23:47:35    |
...
| MIRIAM     | MCKINNEY  | 23:07:08    |
| MINNIE     | ROMERO    | 23:00:34    |
| ELMER      | NOE       | 22:55:13    |
| JEFFERY    | PINSON    | 22:53:33    |
+------------+-----------+-------------+
16 rows in set (0.01 sec)
```
내림차순 정렬은 보통 '상위 5개 계좌의 잔고 표시'와 같은 쿼리 순위를 지정할 때 사용된다.

### 순서를 통한 정렬
`select` 절의 열로 정렬할 때는 이름 대신 `select` 절의 열 나열 순서를 기준으로 열을 참조할 수 있다. 이는 `asc`/`desc` 예제와 마찬가지로 표현식을 정렬할 때 특히 유용하다.

*[`select` 절의 세 번째 요소(열)로 내림차순 정렬을 지정한다.]*
```sql
mysql> SELECT c.first_name, c.last_name,
    ->   time(r.rental_date) rental_date
    -> FROM customer c
    ->   INNER JOIN rental r
    ->   ON c.customer_id = r.customer_id
    -> WHERE date(r.rental_date) = '2005-06-14'
    -> ORDER BY 3 desc;
+------------+-----------+-------------+
| first_name | last_name | rental_date |
+------------+-----------+-------------+
| JEANETTE   | GREENE    | 23:54:46    |
| CHARLES    | KOWALSKI  | 23:54:34    |
| SONIA      | GREGORY   | 23:50:11    |
| TERRENCE   | GUNDERSON | 23:47:35    |
...
| MIRIAM     | MCKINNEY  | 23:07:08    |
| MINNIE     | ROMERO    | 23:00:34    |
| ELMER      | NOE       | 22:55:13    |
| JEFFERY    | PINSON    | 22:53:33    |
+------------+-----------+-------------+
16 rows in set (0.01 sec)
```

`order by` 절의 숫자를 변경하지 않고 `select` 절에 열을 추가하면 예기치 않은 결과가 발생할 수 있으므로 자제한다. 임시 쿼리를 작성할 때는 열을 위치별로 참조할 수 있지만 코드를 작성할 때는 항상 이름으로 열을 참조한다.

# `UPDATE`
```sql
UPDATE [table_name]
SET [field_name1]=[data_value1], 
    [field_name2]=[data_value2], ...
WHERE [field_name]=[data_value]
```

```sql
mysql> UPDATE person
    -> SET street = '1225 Tremont St.',
    -> city = 'Boston',
    -> state = 'MA',
    -> country = 'USA',
    -> postal_code = '02138'
    -> WHERE person_id = 1;
Query OK, 1 row affected (0.02 sec)
Rows matched: 1  Changed: 1  Warnings: 0
```
* `Rows matched: 1` : `where` 절의 조건이 테이블의 단일 행과 일치함
* `Changed: 1` : 표의 단일 행이 수정됨
* `where` 절을 모두 생략하면 `update` 문이 테이블의 모든 행을 수정한다.
  
# `DELETE`
```sql
DELETE FROM [table_name]
WHERE [field_name]=[data_value]
```

```sql
mysql> DELETE FROM person
    -> WHERE person_id = 2;
Query OK, 1 row affected (0.01 sec)
```
`update` 문과 마찬가지로 `where`조건에 따라 둘 이상의 행을 삭제할 수 있으며, `where` 절을 생략하면 모든 행이 삭제된다.
# `use`
```sql
mysql> use sakila;
Database changed
```
`use` 명령어를 통해 작업할 데이터베이스를 지정한다.

# `DROP`
## `DROP TABLE`
```sql
DROP TABLE [table_name]
```

```sql
mysql> DROP TABLE favorite_food;
Query OK, 0 rows affected (0.05 sec)
```


# 내장 `MYSQL` 함수

## `now()`
현재 날짜와 시간을 리턴하는 내장 `MySQL` 함수

```sql
mysql> SELECT now();
+---------------------+
| now()               |
+---------------------+
| 2022-03-08 13:51:26 |
+---------------------+
1 row in set (0.00 sec)
```
## `concat()`
## `upper()`
## `user()`
## `database()`

# `ERROR`
## `Duplicate entry`
```sql
mysql> INSERT INTO person
    -> (person_id, fname, lname, eye_color, birth_date)
    -> VALUES(1, 'Charles', 'Fulton', 'GR', '1968-01-15');
ERROR 1062 (23000): Duplicate entry '1' for key 'person.PRIMARY'
```
테이블 정의에는 기본 키 제약조건 생성이 포함되므로 MySQL은 중복 키 값을 테이블에 삽입하지 않도록 한다. 

## 존재하지 않는 외래 키
`favorite_food` 테이블의 테이블 정의에는 `person_id` 열에 대한 외래 키 제약조건이 있다. 이 제약조건은 `favorite_food` 테이블에 입력된 `person_id`의 모든 값이 `person` 테이블에 존재함을 보증한다.

[`CREATE TABLE favorite_food`](#foreign-key-constraint)
```sql
mysql> INSERT INTO favorite_food (person_id, food)
    -> VALUES(999, 'lasagna');
ERROR 1452 (23000): Cannot add or update a child row: a foreign key constraint fails (`sakila`.`favorite_food`, CONSTRAINT `fk_fav_food_person_id` FOREIGN KEY (`person_id`) REFERENCES `person` (`person_id`))
```

이 경우 `favorite_food` 테이블은 일부 데이터가 `person` 테이블에 의존하므로 `favorite_food` 테이블은 하위(child)로 간주되고 `person` 테이블은 상위(parent)로 간주된다. 두 테이블 모두에 데이터를 입력하려는 경우, 상위인 `person` 테이블에 먼저 관련 행을 작성해야만 `favorite_food` 테이블에 데이터를 입력할 수 있다.

## 열 값 위반
[CREATE TABLE person](#create)

`person` 테이블의 `eye_color` 열은 `'BR'`, `'BL'`, `'GR'` 값으로 제한된다.

열 값을 다른 값으로 잘못 지정하려고 하면 다음과 같은 응답이 나타난다.

```sql
mysql> UPDATE person
    -> SET eye_color = 'ZZ'
    -> WHERE person_id = 1;
ERROR 1265 (01000): Data truncated for column 'eye_color' at row 1
```

## `Incorrect date value`
`date` 열(`date` dtype)을 채울 문자열을 구성할 때 해당 문자열이 예상 형식과 일치하지 않으면 오류가 발생한다.

*[`YYYY-MM-DD`의 기본 날짜 형식과 일치하지 않는 날짜 형식을 사용한 예]*

```sql
mysql> UPDATE person
    -> SET birth_date = 'DEC-21-1980'
    -> WHERE person_id = 1;
ERROR 1292 (22007): Incorrect date value: 'DEC-21-1980' for column 'birth_date' at row 1
```
일반적으로는 기본 형식에 의존하지 않고 **형식 문자열을 명시적으로 지정**하는 편이 좋다.

*[`str_to_date` 함수를 사용하여 사용할 형식 문자열을 지정하는 구문]*
```sql
mysql> UPDATE person
    -> SET birth_date = str_to_date('DEC-21-1980' , '%b-%d-%Y')
    -> WHERE person_id = 1;
Query OK, 1 row affected (0.01 sec)
Rows matched: 1  Changed: 1  Warnings: 0
```

### `string` to `datetime` format

|format|explanation|
|-|-|
|%a| The short weekday name, such as Sun, Mon, ...|
|%b| The short month name, such as Jan, Feb, ...|
|%c| The numeric month (0..12)|
|%d| The numeric day of the month (00..31)|
|%f| The number of microseconds (000000..999999)|
|%H| The hour of the day, in 24-hour format (00..23)|
|%h| The hour of the day, in 12-hour format (01..12)|
|%i| The minutes within the hour (00..59)|
|%j| The day of year (001..366)|
|%M| The full month name (January..December)|
|%m| The numeric month|
|%p| AM or PM|
|%s| The number of seconds (00..59)|
|%W| The full weekday name (Sunday..Saturday)|
|%w| The numeric day of the week (0=Sunday..6=Saturday)|
|%Y| The four-digit year|