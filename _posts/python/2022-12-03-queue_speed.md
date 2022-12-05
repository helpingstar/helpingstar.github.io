---
layout: single
title: "파이썬의 큐 모듈 속도 비교"
date: 2022-12-03 16:51:51
lastmod : 2022-12-03 16:51:53
categories: python
tag: [python, Queue, list, deque]
---

파이썬에는 큐를 구현하는 방법이 세 가지가 있다.

|  | `pop_front` | `push_back` |
|---|---|---|
| `collections.deque` | `popleft()` | `append` |
| `list` | `pop(0)` | `append` |
| `queue.Queue` | `get()` | `put` |

`list`는 기본이고 `queue`는 멀티스레딩에서 사용된다고 한다. `deque`는 `double ended queue`이지만 큐로도 사용할 수 있다.

**실험**

1000번 동안 `push_back`와 `pop`을 반복한다. `push_back`, `pop`은 각가 500번씩 이루어지며 해당 반복을 1000 X 5 번 반복한다.

**실험 전 생각**

`list`가 가장 느리고 `deque`는 양방향으로 구현되어 있기 때문에 그나마 빠르고 `Queue`가 단방향이기 때문에 가장 빠르지 않을까 라는 생각을 했다.

실험 결과는 다음과 같다.

```python
from queue import Queue
from collections import deque
import time

queue_score = []
deque_score = []
list_score = []

queue = Queue()
deq = deque()
lis = list()
```
```python
# deque
start = time.time()
for _ in range(5):
    for _ in range(1000):
        for i in range(1000):
            if i % 2 == 0:
                deq.append(1)
            else:
                deq.popleft()
print(time.time() - start)

>>> 1.0941181182861328
```

```python
# list
start = time.time()
for _ in range(5):
    for _ in range(1000):
        for i in range(1000):
            if i % 2 == 0:
                lis.append(1)
            else:
                lis.pop(0)
print(time.time() - start)

>>> 2.009704351425171
```

```python
# queue
start = time.time()
for _ in range(5):
    for _ in range(1000):
        for i in range(1000):
            if i % 2 == 0:
                queue.put(1)
            else:
                queue.get()
print(time.time() - start)

>>> 13.136821269989014
```

속도 순서는 아래와 같았다.
1. `deque`
2. `list`
3. `Queue`

<!-- TODO deque가 제일 빠른 이유 -->
