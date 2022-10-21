---
layout: single
title: "OpenAI gym 환경 구성시 고려할 점"
date: 2022-10-13 15:17:26
lastmod : 2022-10-13 15:17:29
categories: RL
tag: [env, openai, gym]
toc: true
toc_sticky: true
---

# State

**state의 데이터 형식을 설정하고 그것에 맞춰 함수 알고리즘을 설정할 수도 있고 함수 알고리즘에 맞춰 데이터 형식을 설정할 수도 있다.**

내 경우를 예를 들어보면

![gym_env_pic_1](../../assets/images/rl/gym_env_pic_1.png){: width="50%" height="50%" class="align-center"}

9X9 board가 있고 5x5 배열에 저장된 블록을 board에 놓을 수 있는지 판단해야한다. 그것에 대해 연산하기 위하여 음수를 사용했다. 알고리즘을 짤 때는 `numpy`의 default 형식인 `int32` 기준으로 짰다. 그리고 나중에 적절한 데이터타입에 대한 실험을 하기 위해 `int`, `uint`, `float`들에 대하여 판단해 보았는데, uint에 대해서 casting 오류가 났다.

내 경우는 상태를 표현하는데 0과 1만 사용하기에 별 상관이 없었지만 만약 127~255 사이의 숫자를 사용해야 하는 상황이라면 `uint`를 사용해도 될 것을 `int`를 사용해야 하는 상황이 되는 것이다.

그리고 함수 자체도 음수를 사용하지 않아도 구현 난이도에는 큰 차이가 없었다. 이럴 경우 `uint`를 사용하는 것이 맞다. 내 경우에는 큰 차이가 없어서 그냥 냅두기로 했으나 다음에는 `state`의 데이터 타입에 대한 고려를 하고 `environment`를 구성하기로 했다.
