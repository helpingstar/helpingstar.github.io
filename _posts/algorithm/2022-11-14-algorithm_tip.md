---
title: "알고리즘 PS 오답노트/팁"
date: 2022-11-14 09:27:52
lastmod : 2022-11-14 09:27:54
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
