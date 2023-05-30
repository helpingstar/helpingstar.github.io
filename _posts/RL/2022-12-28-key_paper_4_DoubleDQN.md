---
layout: single
title: Double DQN
date: 2022-12-28 21:55:29
lastmod : 2022-12-28 21:55:27
categories: RL
tag: [RL Paper Review, Double DQN, RL]
toc: true
toc_sticky: true
use_math: true
---

[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), Hasselt et al 2015. Algorithm: Double DQN.

# Paper

## Abstract

Q러닝 알고리즘은 어떤 조건에서 action value를 과대평가한다고 알려져있다. 실제로 그러한 과대평가가 일반적인지, 성과에 해를 끼치는지, 일반적으로 예방될 수 있는지는 알려지지 않았다. 이 논문에서 최근의 DQN 알고리즘이 Atari 게임에서 과대평가로 고통받는 것을 보이고. tabular setting 에서 소개된 Double Q-learning 알고리즘이 큰 규모의 함수근사에서도 일반화될 수 있다는 아이디어를 보인다. 우리는 DQN 알고리즘에 특별한 적용을 할 것을 제안하고 적용된 알고리즘이 가정대로 과대추정을 줄일 뿐만 아니라 몇몇 게임에서 더 나은 결과를 이끌어냈음을 보여준다.

Q러닝 알고리즘은 가끔 비현실적으로 높은 action value를 학습하는 것으로 알려져있다. 왜냐하면 Q러닝 알고리즘은 추정된 action value 중에서 과소평가된 값보다 과대평가된 값을 선호하는 maximization 연산을 포함하기 때문이다.

## Background

### Q-Learning

$$Q_\pi(s, a) \equiv \mathbb{E} \left [ R_1 + \gamma R_2 + \ldots \mid S_0=s, A_0=a, \pi \right ]$$
* $\gamma \in [0, 1]$ : discount factor that trades off the importance of immediate and later rewards.
* optimal value : $Q_{*}(s, a)=\max_{\pi}Q_\pi (s, a)$
* optimal policy : select the highest valued action in each case.

$$\theta_{t+1} = \theta_{t}+\alpha(Y_t^Q-Q(S_t, A_t; \theta_t))\nabla_{\theta_{t}}Q(S_t, A_t; \theta_t) \tag{1}$$

* $\alpha$ : scalar step size

$$Y_t^{Q} \equiv R_{t+1}+\gamma \underset{a}{\max}Q(S_{t+1}, a; \theta_t) \tag{2}$$

### Deep Q Networks

the target network, with parameters $\theta^{-}$, is the same as the online network except that its parameters are copied every $\tau$ steps from the online network, so that then $\theta_t^{-} = \theta_t$, and kept fixed on all other steps. The target used by DQN is then

$$Y_t^{\text{DQN}} \equiv R_{t+1} + \gamma \underset{a}{\max}Q(S_{t+1}, a; \theta_t^{-}) \tag{3}$$

### Double Q-learning

$(2), (3)$에서 보통 Q-learning과 DQN의 max 연산자는 행동의 선택과 평가에서 같은 값을 사용하고 있다. 이것은 과대평가된 행동을 더 선택하게 하고 값 추정도 지나치게 낙관적이게 된다. 이것을 방지하기 위해 평가와 선택을 분리할 수 있는데 이것이 Double Q-leanring의 아이디어이다.

원래의 Double Q-learning 알고리즘에서 두 개의 가치함수가 각각의 경험에 대해서 둘 중 하나가 임의로 업데이트되고  $\theta$, $\theta'$의 두 가중치에서 한 가중치는 greedy policy를 결정하기 위해 사용되고, 하나는 가치를 결정하기 위해 사용된다.

깔끔한 비교를 위해 Q-learning에서 선택과 평가를 분리하고 $(2)$를 다시 작성하면 다음과 같다.

$$Y_t^Q=R_{t+1}+\gamma Q(S_{t+1}, \underset{a}{\argmax}Q(S_{t+1}, a; \theta_t);\theta_t)$$

Double Q-learning error 는 다음과 같이 작성될 수 있다.

$$Y_{t}^{\text{DoubleQ}} R_{t+1}+\gamma Q(S_{t+1}, \underset{a}{\argmax}Q(S_{t+1}, a; \theta_t);\theta'_t) \tag{4}$$

* $\theta_t$ : 현재 가치에 따른 greedy policy 추정
* $\theta'_t$ : 정책의 가치 추정
* $\theta_t, \theta'_t$ 는 두 개의 역할을 서로 바꿔가면서 가중치가 업데이트된다.

## Overoptimism due to estimation errors

Q-learning의 과대추정은 Thrun and Schwartz(1993)에 의해 먼저 조사되었다. 만약 action value가 $\left [ -\epsilon, \epsilon \right ]$ 간격에 균일하게 분포된 무작위 오류가 포함된 경우 각 target은 $\gamma \epsilon \frac{m-1}{m+1}$ 까지 과대평가 된다는 것이다. 여기서 $m$은 action의 개수이다. 그리고 Thun and Schwartz는 이런 과대추정이 점근적으로 sub-optimal policies로 유도하는 것과 함수근사를 사용하는 작은 문제들에서 과대추정이 분명해지는 구체적인 예시를 보였다.

그 뒤에 Van Hasselt(2010)는 환경의 noise가 tabular representation에서도 과대추정이 될 수 있다는 것을 주장했고 Double Q-learning을 해결책으로 제시하였다.

이 섹션에서는 어떠한 종류의 추정오류도 environmental noise, 함수근사, 비정상성 그외 어느 원인이든 상향 편향(upward bias)을 일으킬 수 있음을 보다 일반적으로 보여준다. 이것은 중요한데, 왜냐하면 실제로 어느 방법도 실제 값은 알 수 없기 때문에 학습중에 부정확성이 일어날 수 일어나기 때문이다.

위에서 인용한 Thrun and Schwartz(1993)의 결과는 구체적인 상황에서의 과대추정의 상한선을 제시하지만, 하한선을 유도하는 것도 가능하다.

### Theorem 1.

모든 treu optimal action value가 같은
