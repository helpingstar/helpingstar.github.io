---
layout: single
title: "SQL 자료형 복습"
date: 2022-03-08 15:24:30
lastmod : 
categories: sql
tag: [sql]
toc: true
toc_sticky: true
use_math: true
---
# 문자 데이터
## 고정 길이 문자열
```sql
char(20) /* fixed-length */
```
공백으로 오른쪽이 채워지고 항상 동일한 수의 바이트를 사용

`char`열의 최대 길이는 현재 255바이트

열에 저장할 모든 문자열이 약어처럼 길이가 동일할 때 사용
## 가변 길이 문자열
```sql
varchar(20) /* variable-length */
```
공백으로 오른쪽이 채워지지 않고 항상 동일한 수의 바이트를 사용하지 않는다.

`varchar` 열은 최대 65,535바이트까지 사용할 수 있다.

열에 저장할 문자열의 길이가 서로 다를 때 사용

## 캐릭터셋
각 문자마다 여러 바이트의 저장공간이 필요한데, 이러한 문자 집합, 즉 캐릭터셋을 **멀티바이트캐릭터셋**이라고 한다.
MySQL은 싱글 바이트 및 멀티 바이트의 다양한 캐릭터셋을 모두 사용하여 데이터를 저장할 수 있다. 

```SQL
mysql> SHOW CHARACTER SET;
+----------+------------------------- +---------------------+--------+
| Charset  | Description              | Default collation   | Maxlen |
+----------+------------------------- +---------------------+--------+
| armscii8 | ARMSCII-8 Armenian       | armscii8_general_ci |      1 |
| ascii    | US ASCII                 | ascii_general_ci    |      1 |
| big5     | Big5 Traditional Chinese | big5_chinese_ci     |      2 |
| binary   | Binary pseudo charset    | binary              |      1 |
| cp1250   | Windows Central European | cp1250_general_ci   |      1 |
| cp1251   | Windows Cyrillic         | cp1251_general_ci   |      1 |
| cp1256   | Windows Arabic           | cp1256_general_ci   |      1 |
| cp1257   | Windows Baltic           | cp1257_general_ci   |      1 |
...
41 rows in set (0.00 sec)
```

`show` 명령어를 사용하면 서버에서 지원되는 캐릭터셋을 볼 수 있다.

네 번째 열 `maxlen`의 값이 `1`보다 크면 캐릭터셋이 멀티 바이트 캐릭터셋이라는 의미이다.

MYSQL 8.0은 기본 캐릭터셋으로 `utf8mb4`를 적용한다. 

열을 정의할 때 기본값이 아닌 캐릭터셋을 선택하려면 다음과 같이 자료형 정의 뒤에 지원되는 캐릭터셋 중 하나를 지정한다.
```sql
varchar(20) character set latin1
```

MYSQL을 사용하면 전체 데이터베이스에 대한 기본 캐릭터셋을 설정할 수 있다.
```sql
create database european_sales character set latin1;
```
## 텍스트 데이터
`varchar` 열에 64KB를 초과하는 데이터를 저장하려면 텍스트 자료형 중 하나를 사용해야 한다.

|자료형|최대 바이트 크기|
|-|-|
|`tinytext`|`255`|
|`text`|`65,535`|
|`mediumtext`|`16,777,215`|
|`longtext`|`4,294,967,295`|

텍스트 자료형중 하나 선택시 고려할 것
* 텍스트 열에 로드되는 데이터가 해당 유형의 최대 크기를 초과하면 데이터가 잘림
* 데이터를 열에 로드되면 후행 공백이 제거되지 않음
* 정렬 또는 그룹화에 `text` 열을 사용할 경우, 필요하다면 한도를 늘릴 수 있지만 처음에는 1,024바이트만 사용된다.
* `text`를 제외한 텍스트 자료형은 MYSQL의 고유한 자료형이다. SQL 서버에는 큰 문자 데이터에 대한 단일 `text` 자료형이 있지만, DB2와 오라클은 큰 문자 오브젝트에 `clob`이라는 자료형을 사용한다.
* MYSQL은 이제 `varchar` 열에 최대 65,535바이트를 허용하므로 `tinytext`나 `text` 자료형을 사용할 필요가 없다.
  
### ex)
* `varchar` : 고객과의 상담 내역을 저장하기 위해 자유 형식의 데이터 입력용 열


* `mediumtext`, `longtext` : 문서를 저장

# 숫자 데이터
* 고객 주문의 배송 여부를 나타내는 열
  * 불리언 : 0(false) / 1(true)
* 트랜잭션 테이블의 시스템 생성 기본 키
  * 보통 1에서 시작하여 잠재적으로 매우 큰 수까지 1씩 증가
* 고객의 온라인 장바구니 품목 번호
  * 1과 200 사이의 양의 정수
* 회로 기판 드릴 기계의 위치 데이터
  * 소수점 8자리까지 정확도를 요구

## MYSQL 정수 자료형

|dtype|Signed range|Unsigned range|
|-|-|-|
|`tinyint`|-128 to 127|0 to 255|
|`smallint`|-32,768 to 32,767|0 to 65,535|
|`mediumint`|-8,388,608 to 8,388,607|0 to 16,777,215|
|`int`|-2,147,483,648 to 2,147,483,647|0 to 4,294,967,295|
|`bigint`|$-2^{63}$ to $2^{63}$ -1|0 to $2^{64}$-1|


## MYSQL 부동소수점 자료형

|Type|Numeric range|
|-|-|
|`float(p,s)`|−3.402823466E+38 to −1.175494351E-38 <br>1.175494351E-38 to 3.402823466E+38|
|`double(p,s)`|−1.7976931348623157E+308 to −2.2250738585072014E-308 <br>2.2250738585072014E-308 to 1.7976931348623157E+308|

* `p` : 정밀도(precision), 자릿수
* `s` : 척도(scale), 소수점 아래 자릿수
* `p`, `s`를 지정할 수 있지만 필수는 아니다.
* 정밀도, 척도를 초과하면 열에 저장된 데이터는 반올림된다.
  * ex) `float(4, 2)` : `17.8675` -> `17.87`, `178.375` -> `error`

부동 소수점의 `unsigned` : 데이터의 범위를 변경하는 대신 음수가 저장되는 것을 방지하는 역할

# 시간 데이터

**MySQL 시간 자료형**

|Type|Default format|Allowable values|
|-|-|-|
|`date`|YYYY-MM-DD|1000-01-01 to 9999-12-31|
|`datetime`|YYYY-MM-DD HH:MI:SS|1000-01-01 00:00:00.000000 <br>to 9999-12-31 23:59:59.999999|
|`timestamp`|YYYY-MM-DD HH:MI:SS|1970-01-01 00:00:00.000000 <br>to 2038-01-18 22:14:07.999999|
|`year`|YYYY|1901 to 2155|
|`time`|HHH:MI:SS|−838:59:59.000000 <br>to 838:59:59.000000|

`datetime`, `timestamp`, `time` : 0에서 6 사이의 값을 제공하여 소수점 이하 6자리(마이크로초) 까지 사용할 수 있다.

ex) `datetime(2)` : 시간 값이 100분의 1초를 포함한다.

**날짜 형식의 구성 요소**

|Component|Definition|Range|
|-|-|-|
|`YYYY`|Year, including century|1000 to 9999|
|`MM`|Month|01(January) to 12(December)|
|`DD`|Day|01 to 31|
|`HH` |Hour| 00 to 23|
|`HHH` |Hours| (elapsed) −838 to 838|
|`MI` |Minute| 00 to 59|
|`SS` |Second| 00 to 59|

**`timestamp` vs `datetime`**

테이블에 행이 추가되거나 수정될 때 MySQL 서버에 의해 현재 날짜/시간으로 `timestamp`열이 자동으로 채워진다, `timestamp`는 `time_zone`에 의존한다.