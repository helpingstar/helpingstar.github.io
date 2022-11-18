---
title: "알고리즘 PS 오답노트/팁"
date: 2022-11-14 09:27:52
lastmod : 2022-11-18 18:03:05
categories: algorithm
tag: [algorithm, ps]
toc: true
toc_sticky: true
use_math: true
---

## 1.

dfs로 재귀적 탐색을 한다면 주어진 조건을 충족시키지 못한다는 것이 명백하거나, 주어진 조건을 충족했다면 함수를 종료해야한다.

 [**BOJ 1941**](https://www.acmicpc.net/problem/1941) 문제에서 학생 수가 7명이 되면 탐색을 끝내야 하는데 학생수 7명을 체크하고 숫자를 하나 늘린다음에 함수를 끝내지 않아 8, 9, 10... 까지 계속 탐색하게 되어 시간초과가 일어났다.

함수를 종료시키는 방법으로는 `if True ~ else: ~` 또는 `if True return else: ~` 방법이 있을 수 있다.


## 2.

그래프 문제(ex. 다익스트라)를 풀 때 `INF` 설정에 주의해야 한다. [**BOJ 9370**](https://www.acmicpc.net/problem/9370) 문제에서 `INF`값으로 `distance`를 초기화하고 문제를 풀기 위해 `distance`를 도로의 길이의 최댓값인 1000에 1을 더해 1001로 했다. 이러면 안된다. 어떤 두 점 사이의 최대 길이는 (노드의 개수-1) X (도로의 최대 길이)가 되어야 한다. 순간 접한 두 점사이의 거리의 최댓값과 임의의 두 점 사이의 거리를 헷갈렸다. 효율적이지는 않지만 헷갈린다면 `sys.maxsize`를 쓰도록 하자.

## 3. operator chaining
```python
>>> one = [1]
>>> 1 in one != 2 in one
False
>>> (1 in one) != (2 in one)
True
>>> ((1 in one) != 2) in one
True
```

문제의 조건을 판단하는 문제에서 첫번째와 같이 작성하여 특정 조건에 진입하지 못하게 되었다. 내가 가진 의도는 두 번째 처럼 계산되는 것이었다.

연산자 우선순위 문제인가 생각이 들어서 두번째, 세번째처럼 작성하였으나 모두 `True`를 반환하였다. (3번째 것도 `True in one`이어서 `True`를 반환하여 의도와 맞지 않기는 하다.)

파이썬 공식 문서를 보자.

![nesterov_gradient](/assets/images/python/comparison_chaining.png){: width="80%" height="80%" class="align-center"}

정리하면 다음과 같다

파이썬은 C와 달리 `a < b < c` 사용이 가능하다

비교연산은 chain될 수 있는데 예를 들어 `x < y <= z`는 `x < y and y <= z`의 의미를 갖는다. 이를 `comparison chaining`이라고 한다.

확장해서 `a op1 b op2 c ... y opN z`는 `a op1 b and b op2 c and ... y opN z`의 의미를 갖고 이는 한번만 계산된다. (`a op1 b`를 하고 `a op2 c`를 또하지 않는다는 이야기다)

문제로 돌아가보자 그럼 내 식이었던
```python
1 in one != 2 in one
```
은

```python
1 in one and one != 2 and 2 in one
# == (1 in one) and (one != 2) and (2 in one)
```
과 같은 의미가 된 것이고 `2 in one == False`이기 때문에 `False`를 반환한 것이다.

## 4. `sort(key=??)`

파이썬은 정렬시에 `key`에 함수를 인자로 줄 경우 조건 외 부분은 기존의 순서를 유지한다. 기본 정렬이 아닐 경우 주의해야 한다.

예를 보자
```python
>>> test = [[2, 3], [2, 2], [3, 2], [3, 1]]
>>> sorted(test)
[[2, 2], [2, 3], [3, 1], [3, 2]]
```
`key`에 아무것도 설정하지 않으면 생각하는 기본 순서대로 우선순위를 왼쪽에서 오른쪽으로 가면서 정렬한다.

다음 예를 보자
```python
>>> test = [[2, 3], [3, 2], [2, 2], [3, 1]]
>>> sorted(test, key=lambda x: x[1])
[[3, 1], [3, 2], [2, 2], [2, 3]]
```
두번째 원소를 기준으로 정렬했다. 그런데 `[3, 2]`와 `[2, 2]`는 뒤에서부터 정렬하면 `[2, 2]`가 앞으로 나와야할 것 같지만 `key`에 들어간 함수로 정렬할 때는 요소 외의 이외 부분은 신경도 쓰지 않고 기존의 순서 (`[3, 2], [2, 2]`)를 유지한다.

만약 뒤에부터 정렬하고싶다면

```python
>>> test = [[2, 3], [3, 2], [2, 2], [3, 1]]
>>> sorted(test, key=lambda x: x[::-1])
[[3, 1], [2, 2], [3, 2], [2, 3]]
```
처럼 쓰면 될 것이다.

## 5. `Dynamic Programming`
아주 간단한 문제이다. 문제가 DP로 풀린다는 느낌이 올 경우 재귀코드로 헷갈리게 생각하지말고 일단 수학적으로 점화식을 떠올린 후에 그것을 코드로 옮기자.

## 6. Traveling Sales Person

```python
import sys

input = sys.stdin.readline

n_city = int(input())
graph = [list(map(int, input().split())) for _ in range(n_city)]
INF = int(1e9)

board = [[0] * (1 << n_city) for _ in range(n_city)]

def tsp(now, visited):
    if visited == (1 << n_city) - 1:
        if graph[now][0]:
            return graph[now][0]
        else:
            return INF

    if board[now][visited]:
        return board[now][visited]

    n_min = INF
    for next in range(1, n_city):
        if not graph[now][next]:
            continue
        if visited & (1 << next):
            continue
        n_min = min(n_min, graph[now][next]
                    + tsp(next, visited | 1 << next))
    board[now][visited] = n_min

    return board[now][visited]

print(tsp(0, 1))
```
`TSP` 문제에서 모두 방문한 경우 마지막에 `now`에서 `0`으로가는 길을 설정할 때 길이 0이면 `INF`를 리턴해야 한다. 그래야 그것을 받은 호출자에서 그것을 최단경로로 선택하지 않는다. 나는 그러지 않고 그냥 반환했다가 길이 없음을 의미하는 0을 0으로 반환해서 그것이 선택되게 해서 틀렸다.

또 초기 `board`를 `INF`보다는 `0`으로 초기화하는 것이 빠른 것 같다. 뒤에서 조건판단할때는 상관이 없다. 그런데 `INF`로 초기화하면 시간초과가 났다. 그래서 실험을 해보았다.
