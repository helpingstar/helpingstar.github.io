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

# `SELECT`
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
```sql
mysql> SELECT now();
+---------------------+
| now()               |
+---------------------+
| 2022-03-08 13:51:26 |
+---------------------+
1 row in set (0.00 sec)
```
`now()` : 현재 날짜와 시간을 리턴하는 내장 `MySQL` 함수


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