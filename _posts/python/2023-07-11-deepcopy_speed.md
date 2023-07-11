---
layout: single
title: "Python copy.deepcopy vs list[:]"
date: 2023-07-11 09:52:27
lastmod : 2023-07-11 09:52:27
categories: python
tag: [python, deepcopy]
use_math: true
---

파이썬에서 리스트를 깊은 복사하는 방법에는 두 가지가 있다.

첫 번째는 `copy.deepcopy` 를 사용하는 것이고 두 번째는 `list[:]` 와 같이 인덱스 슬라이스를 비우는 방법이다. 물론 일차원에 대해서만 허용되지만 이 두가지의 속도를 비교해보자

(`timeit`의 실행/반복횟수는 생략)

코드 아래에 표로 보기 쉽게 정리하였다.
```python
# 길이 10의 1차원 배열
length = 10
source = [i for i in range(length)]

%timeit dest1 = copy.deepcopy(source)
>>> 5.62 µs ± 84.8 ns per loop
%timeit dest2 = source[:]
>>> 286 ns ± 140 ns per loop

# 길이 100의 1차원 배열
length = 100
source = [i for i in range(length)]

%timeit dest3 = copy.deepcopy(source)
>>> 45.1 µs ± 1.63 µs per loop
%timeit dest4 = source[:]
>>> 319 ns ± 7.08 ns per loop

# 길이 10000의 1차원 배열
length = 10000
source = [i for i in range(length)]

%timeit dest5 = copy.deepcopy(source)
6.36 ms ± 2 ms per loop
%timeit dest6 = source[:]
26.9 µs ± 487 ns per loop

```

| length | `copy.deepcopy` | `list[:]` |
|--------|-----------------|-----------|
| 10     | 5.62 µs         | 286 ns    |
| 100    | 45.1 µs         | 319 ns    |
| 10000  | 6.36 ms         | 26.9 µs   |

|  Base 10  | Symbol (name) |
|-----------|---------------|
| $10^{−3}$ | m (milli)     |
| $10^{−6}$ | µ (micro)     |
| $10^{−9}$ | n (nano)      |

보다시피 `list[:]`가 훨씬 빠르다. 내 생각보다 훨씬 더 빨랐다.

그럼 2차원일 경우에는 어떨까? 간단하게 실험해보았다.

```python
L = 1000
# 1000 X 1000의 2차원 배열 선언
test = [[i for i in range(L)] for j in range(L)]

test2 = copy.deepcopy(test)
test3 = []
for i in range(L):
    test3.append(test[i][:])

# 동일함 확인
test2 == test3
>>> True

# copy.deepcopy
%%timeit
test2 = copy.deepcopy(test)

>>> 432 ms ± 11.7 ms per loop

# iteration + list[:]
%%timeit

test3 = []
for i in range(L):
    test3.append(test[i][:])

>>> 11.9 ms ± 1.24 ms per loop
```

더 이상 실험은 안하지만 1000 X 1000 배열에 대해 40배 정도 속도 차이가 났다. test3에 대한 코드가 그리 깔끔하지 않은데도 말이다.

앞으로 `copy.deepcopy`를 사용하게 될 경우 꼭 써야 할 상황인지를 확인해야겠다.
