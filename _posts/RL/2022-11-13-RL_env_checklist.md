---
title: "강화학습 환경 만들기 체크리스트"
date: 2022-11-13 16:13:50
lastmod : 2022-11-13 16:13:53
categories: RL
tag: [RL, gym]
toc: true
toc_sticky: true
use_math: true
---

# space
* `observation/action_space`
  * 형식을 정한다.
    * `Box`, `Discrete`, ...
  * 데이터타입을 정한다
    * `uint8`, `float32`, ...
  * 랭크를 결정한다.
    * 2차원 배열을 `[1, m, n]`으로 반환할지, `[m, n]`으로 반환할 지 등

# State

# Action

# Reward
* `reward`를 실제 구현하려는 대상과 같게 할지 (ex. 게임의 점수) 조정할지를 선택한다 (ex.[1, -1])
