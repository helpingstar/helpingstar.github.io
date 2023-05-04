---
title: "알고리즘 PS 오답노트/팁 2"
date: 2023-05-04 23:51:31
lastmod : 2023-05-04 23:51:34
categories: algorithm
tag: [algorithm]
toc: true
toc_sticky: true
use_math: true
---

앞 내용(1~20)은 ["알고리즘 PS 오답노트/팁"](https://helpingstar.github.io/algorithm/algorithm_tip/)에서 찾아볼 수 있다.

## 21.

구현에서 어떤 조건의 만족 여부, 개수를 찾는 문제라면 **문제를 만족하는 경우의 수를 제대로 파악해야 한다.**

[**BOJ 2615**](https://www.acmicpc.net/problem/2615)

오목의 달성 여부를 판단하는 문제인데 좌상단에서 우하단으로 찾는아서 경우의 수를 줄인다는 것에 매몰되어 가로(→), 세로(↓), 우하 대각선(↘) 만 파악해서 문제를 풀지 못했다. 실제로는 우상 대각선(↗)도 존재한다.