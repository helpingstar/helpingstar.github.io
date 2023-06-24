---
layout: single
title: "Python bisect.insort vs list.sort"
date: 2023-06-25 00:28:23
lastmod : 2023-06-25 00:28:23
categories: python
tag: [python, bisect, insort, sort]
toc: true
toc_sticky: true
---

파이썬 표준 라이브러리에는 이진 탐색을 위한 `bisect` 가 있다.

코딩 테스트시에 주로 `bisect.bisect_xxx`을 활용하여 이분탐색을 하는데 활용하지만 이 외에도 `bisect.insort`가 있다.

어디에 사용하는 것일까? 공식문서를 보자

---

**`bisect.insort(a, x, lo=0, hi=len(a), *, key=None)`**

* `a` : list
* `x` : item

Similar to `insort_left()`, but inserting x in a after any existing entries of x.

This function first runs `bisect_right()` to locate an insertion point. Next, it runs the `insert()` method on a to insert x at the appropriate position to maintain sort order.

...

---

`bisect_right()`를 실행하여 삽입 위치를 찾은 후 `insert()` 함수를 통해 해당 인덱스에 `x`를 삽입한다.

cpython을 통해 함수를 보자

```python
# cpython/Lib/bisect.py : 118

insort = insort_right

# cpython/Lib/bisect.py : 4~18
def insort_right(a, x, lo=0, hi=None, *, key=None):
    ...
    if key is None:
        lo = bisect_right(a, x, lo, hi)
    else:
        lo = bisect_right(a, key(x), lo, hi, key=key)
    a.insert(lo, x)
...
```

그럼 이 함수는 어떤 상황에 사용하면 좋을까? 리스트의 내장 함수인 `list.sort()`를 사용하면 안되는 것일까??

# `bisect.insort()` vs `list.sort()`

당연하다고 느낄 수 있는 것들이지만 이제부터 실험해보겠다.

**요소들을 모두 집어넣고 마지막에 정렬된 상태를 이용할 때**

```python
numbers = np.random.randint(0, 50000, size=(5000,))

# 요소들을 정렬된 상태로 유지하면서 요소를 삽입한다.
def test1():
    L = []
    for i in range(5000):
        bisect.insort(L, numbers[i])

# 요소들을 모두 집어넣은 후 마지막에 정렬한다.
def test2():
    L = []
    for i in range(5000):
        L.append(numbers[i])
    L.sort()

%timeit test1()
>>> 9.58 ms ± 355 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit test2()
>>> 3.06 ms ± 26.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

두번째가 훨씬 빠르다. 요소들을 모두 삽입한 후에 정렬된 상태를 이용하려면 굳이 중간마다 `bisect.insort()`를 할 필요가 없다.

**요소들을 집어넣으면서 중간 중간에 정렬된 상태가 필요할 때**

```python
# 요소들을 삽입할 때마다 sort 함수를 이용하여 정렬한다.
def test3():
    L = []
    for i in range(5000):
        L.append(numbers[i])
        L.sort()

%timeit test3()
>>> 291 ms ± 7.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

바로바로 정렬하는 것이 압도적으로 느리다. 바로바로 정렬해서 그렇다고 할 수 있다. 정렬의 빈도수를 줄여서 테스트해보자.

```python
# 요소들을 10개 삽입할 때마다 sort 함수를 이용하여 정렬한다.
def test4():
    L = []
    for i in range(5000):
        L.append(numbers[i])
        if i % 10 == 0:
            L.sort()


# 요소들을 100개 삽입할 때마다 sort 함수를 이용하여 정렬한다.
def test5():
    L = []
    for i in range(5000):
        L.append(numbers[i])
        if i % 100 == 0:
            L.sort()

%timeit test4()
>>> 32.5 ms ± 351 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit test5()
>>> 6.43 ms ± 47.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
10개 삽입할 때마다 정렬시에는 더 느렸지만 100개 삽입할 때마다 정렬할 시에는 `sort`가 더 빨랐다. (`insort` : 9.58 ms)

그래서 숫자를 더 늘리고 줄여서 실험해 보았다. 조건문 때문에 실험에 지장이 있을 수 있으니 `insort`에 조건문을 임의로 넣어서 실험했다.

```python
# 요소의 개수를 늘린다.
numbers = np.random.randint(0, 100000, size=(10000,))

def test6():
    L = []
    for i in range(10000):
        if i % 100 == 0:
            pass
        bisect.insort(L, numbers[i])

def test7():
    L = []
    for i in range(10000):
        L.append(numbers[i])
        if i % 100 == 0:
            L.sort()

%timeit test6()
>>> 34.3 ms ± 1.6 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit test6()
>>> 20.4 ms ± 632 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```
요소의 개수를 늘렸을 때 `list.sort()`가 빠르다.

```python
numbers = np.random.randint(0, 100000, size=(1000,))

def test8():
    L = []
    for i in range(1000):
        if i % 100 == 0:
            pass
        bisect.insort(L, numbers[i])

def test9():
    L = []
    for i in range(1000):
        L.append(numbers[i])
        if i % 100 == 0:
            L.sort()

%timeit test7()
>>> 942 µs ± 42.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit test8()
>>> 680 µs ± 39.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```
요소의 개수를 줄여도 `list.sort()`가 빠르다.

# 결론

* 요소를 다 넣고 마지막에 정렬하면 될 때 : `list.sort()`

* 요소를 넣으면서 중간 중간에 어떤 추가 처리를 할 때:
  * 넣을 때마다 추가 처리 : `insort()`
  * 그외 : 추가 처리 빈도가 높을 수록 `insort()`가 유리하고 빈도가 낮을 수록 `list.sort()`가 유리하다.
    * 간단한 실험으로는 10번마다 정렬시 `insort()`가 유리하고 100번마다 정렬시 `list.sort()`가 유리하다.
