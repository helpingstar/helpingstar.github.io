---
layout: single
title: "파이썬 리스트 선언 속도 비교"
date: 2023-07-11 10:00:49
lastmod : 2023-07-11 10:00:49
categories: python
tag: [python, list, "list comprehension"]
use_math: true
---

파이썬 리스트 선언 방법에는 여러 가지가 있다. 실험을 통해 각각의 속도를 비교해보자

각 방법은 다음과 같다.
1. `list(range)`
2. `[i for i in range]`
3. `list(i for i in range)`
4. `list.append(x)`

아래 코드와 같이 실험을 진행했으며 `L`을 각각 다르게 하여 실험하였다.
```python
# 1
%timeit test = list(range(L))
# 2
%timeit test = [i for i in range(L)]
# 3
%timeit test = list(i for i in range(L))
# 4
%%timeit
test = []
for i in range(L):
    test.append(i)
```

|  Base 10  | Symbol (name) |
|-----------|---------------|
| $10^{−3}$ | m (milli)     |
| $10^{−6}$ | µ (micro)     |
| $10^{−9}$ | n (nano)      |

| L      | 1       | 2       | 3         | 4       |
|--------|---------|---------|-----------|---------|
| 100    | 862 ns  | 3.13 µs | 7.11 µs   | 5.72 µs |
| 1000   | 16.9 µs | 34.2 µs | 66.3   µs | 60 µs   |
| 10000  | 192 µs  | 588 µs  | 511 µs    | 947 µs  |
| 100000 | 2.77 ms | 4.09 ms | 5.99 ms   | 8.77 ms |

* ~1000: **1 > 2 > 4 > 3**
* 10000: **1 > 3 > 2 > 4**
* 100000: **1 > 2 > 3 > 4**

순으로 빨랐다. 사실 3번 코드는 잘 쓰이지 않는다는 것을 감안하면 1번 코드를 쓰는게 가장 좋고 4번의 형태는 쓰지 않는 것이 좋겠다. 2번의 경우에는 위에 적지는 않았지만 분산이 가장 높았다. 이건 특이한 경우라고 볼 수 있겠다.

숫자가 적을 때 3번 코드가 느린 것은 제너레이터가 생성되는 것이 시간이 걸리는 것으로 추정된다.
