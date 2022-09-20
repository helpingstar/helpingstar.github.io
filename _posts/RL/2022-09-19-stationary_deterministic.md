---
layout: single
title: "강화학습에서 stationary와 deterministic"
date: 2022-09-19 20:40:02
lastmod : 2022-09-19 20:39:55
categories: RL
tag: [RL, stationary, deterministic]
toc: true
toc_sticky: true
use_math: true
---

스터디 중 `stationary` 관련하여 설명하였다, 그런데 설명 후 그것은 `deterministic`이라는 지적을 받았다. 생각해보니 그 두개를 혼재하며 사용하고 있었다는 것을 느껴 관련하여 정리하기 위하여 이 글을 쓴다.


# deterministic

https://ai.stackexchange.com/a/12275/46808

*A deterministic policy is a function of the form $\pi_d:S \rightarrow A$, that is, a function from the set of states of the environment, $S$, to the set of actions, $A$. The subscript $d$ only indicates that this is a deterministic policy.*

deterministic policy는 환경의 집합 $S$와 action의 집합 $A$에서 $S \rightarrow A$의 형태를 가지는 함수이다. 아래에 기입한 $d$는 deterministic policy를 의미한다.

*For example, in a grid world, the set of states of the environment, $S$, is composed of each cell of the grid, and the set of actions, $A$, is composed of the actions "left", "right", "up" and "down". Given a state $s \in S$, $\pi(s)$ is, with probability $1$, always the same action (e.g. "up"), unless the policy changes.*

예를 들어, 각 격자 셀로 구성된 환경의 `state` 집합 $S$와 상,하,좌,우로 구성된 `action` 집합 $A$로 구성된 grid world가 있다. `state`의 집합 $S$의 한 요소 $s$에 대해 $\pi(s)$는 `policy`가 바뀌지 않는 한 1의 확률로 언제나 같은 `action`을 얻는다.

*A stochastic policy can be represented as a family of conditional probability distributions, $\pi_s(A \mid S)$, from the set of states, $S$, to the set of actions, $A$. A probability distribution is a function that assigns a probability for each event (in this case, the events are actions in certain states) and such that the sum of all the probabilities is 1.*

`stochastic policy`는 $S$가 `state`의 집합이고 $A$가 `action`의 집합일 때 $\pi_s(A \mid S)$로 나타내며, 조건부 확률 분포의 한 계열로 표현될 수 있다.,

*A stochastic policy is a family and not just one conditional probability distribution because, for a fixed state $s \in S$, $\pi_s(A \mid S=s)$ is a possibly distinct conditional probability distribution. In other words, $\pi_s(A \mid S=s)=\pi_s(A \mid S=s_1),...,\pi_s(A \mid S=s_{\vert S \vert})$, where $\pi_s(A \mid S=s)$ is a conditional probability distribution over actions given that the state is $s \in S$ and $\mid S \mid$ is the size of the set of states of the environment.*

`stochastic policy`는 하나의 조건부 확률분포가 아닌 집합이다. 왜냐하면 고정된 상태 $s \in S$에서 $\pi_s(A \mid S=s)$는 아마도 별개의 조건부 확률 분포이기 때문이다. 다른 말로 $\pi_s(A \mid S=s)=\{\pi_s(A \mid S=s_1),...,\pi_s(A \mid S=s_{\vert S \vert})\}$에서 $\pi_s(A \mid S=s)$는 `state`가 $s \in S$이고 집합의 크기가 $\mid S \mid$인 경우 `action`에 대한 조건부 확률 분포이다.

*Often, in the reinforcement learning context, a stochastic policy is misleadingly (at least in my opinion) denoted by $π_s(a∣s)$, where $a∈A$ and $s∈S$ are respectively a specific action and state, so $π_s(a∣s)$ is just a number and not a conditional probability distribution.*

자주 강화학습의 문맥에서, `stochastic policy`는 (적어도 나의 의견으로는) $\pi_s(a \mid s)$로 잘못 표기되는 경우가 있다. 여기서 $a \in A$와 $s \in S$는 각각 특정 `action`과 `state`이므로 $\pi_s(a \mid s)$는 그저 숫자이며 조건부 확률 분포가 아니다.

*A single conditional probability distribution can be denoted by $π_s(A∣S=s)$, for some fixed state $s∈S$. However, $π_s(a∣s)$ can also denote a family of conditional probability distributions, that is, $π_s(A∣S)=π_s(a∣s)$, if $a$ and $s$ are arbitrary.*

고정된 `state` $s \in S$에서 단일 조건부 확률 분포는 $\pi_s(A \mid S=s)$로 표기될 수 있다. 그러나 $a$와 $s$가 임의인 경우 $\pi_s(a \mid s)$ 또한 조건부 확률 분포의 계열, $\pi_s(A \mid S)=\pi_s(a \mid s)$로 표기될 수 있다.

*Alternatively, $a$ and $s$ in $\pi_s(a \mid s)$ are just (dummy or input) variables of the function $\pi_s(a \mid s)$ (i.e. p.m.f. or p.d.f.): this is probably the most sensible way of interpreting $\pi_s(a \mid s)$ when you see this notation (see also this answer).*

또는, $a$와 $s$는 함수 $\pi_s(a \mid s)$(즉, p.m.f., p.d.f.)의 (dummy또는 입력) 변수일 뿐이다. 이것은 아마도 이 표기법을 볼 때 $\pi_s(a \mid s)$를 해석하는 가장 합리적인 방법일 것이다(이 답변도 마찬가지)

*In this case, you could also think of a stochastic policy as a function $\pi_s:S \times A \rightarrow [0,1]$, but, in my view, although this may be the way you implement a stochastic policy in practice, this notation is misleading, as the action is not conceptually an input to the stochastic policy but rather an output (but in the end this is also just an interpretation).*

여기에서 너는 `stochastic policy`를 함수 $\pi_s:S \times A \rightarrow [0,1]$로 생각할 수도 있다. 그러나 내 생각은 비록 이것이 당신이 실제로 `stochastic policy`를 구현하는 방법일 수도 있지만 이 표기법은 오류가 있다. 왜냐하면 `action`은 개념적으로 `stochastic policy`에 대한 input이 아니라 output이기 때문이다.(그러나 결국 이것 또한 단순한 해석일 뿐이다.)

*In the particular case of games of chance (e.g. poker), where there are sources of randomness, a deterministic policy might not always be appropriate.*

randomness 의 source가 있는 확률 게임의 특별한 경우(예: 포커), `deterministic policy`는 항상 적절한 것은 아니다


*For example, in poker, not all information (e.g. the cards of the other players) is available. In those circumstances, the agent might decide to play differently depending on the round (time step).*

예를 들어, 포커에서 모든 정보가 접근 불가능하다(상대방의 카드). 그런 환경에서 `agent`는 라운드(time step)에 따라 다르게 게임을 할 수 있도록 판단을 할 것이다.

*More concretely, the agent could decide to go "all-in" 23 of the times whenever it has a hand with two aces and there are two uncovered aces on the table and decide to just "raise" 13 of the other times.*

더 구체적으로, 그의 손에 두개의 ace가 있고 테이블에 ace 두개가 공개되어 있을 때 23번은 'all-in'을 결정하고 나머지 경우는 13번의 'raise'를 하기로 결정할 수도 있다.

*A deterministic policy can be interpreted as a stochastic policy that gives the probability of 1 to one of the available actions (and 0 to the remaining actions), for each state.*

`deterministic policy`는 각 `state`에 대해 가능한 `action` 중 하나(그리고 나머지 행동에는 0)의 확률을 주는 확률적 정책으로 해석될 수 있다.

---

https://ai.stackexchange.com/a/20960/46808

**Deterministic Policy** :

Its means that for every state you have clear defined action you will take

For Example: We 100% know we will take action **A** from state **X**.

모든 `state`에 대해 명확하게 정의된 `action`이 있다는 것을 의미한다.

예: `state` **X**에서 `action` **A**를 할 것임을 100% 알 수 있다.

**Stochastic Policy** :

Its mean that for every state you do not have clear defined action to take but you have probability distribution for actions to take from that state.

모든 `state`에 대해서 명확하게 정의된 `action`이 없으나 `state` 에서 취할 `action`에 대한 확률 분포를 가지고 있다는 것을 의미한다.

For example there are 10% chance of taking action **A** from state **S**, There are 20% chance of taking **B** from State **S** and there are 70% chance of taking action **C** from state **S**, Its mean we don't have clear defined action to take but we have some probability of taking actions.

예를 들어 **S**에서 각각 10%, 20%, 70%의 확률로 **A, B, C**를 할 확률을 가진다고 하면, 명확하게 정의된 `action`은 없으나 `action`을 취하는 것에 대한 몇개의 확률을 가지고 있음을 의미한다.

# stationary

https://ai.stackexchange.com/a/15429/46808

***A stationary policy**, $\pi_t$, is a policy that does not change over time, that is, $\pi_t=\pi, \forall t \ge 0$, where $\pi$ can either be a function, $\pi : S \rightarrow A$ (a deterministic policy), or a conditional density, $\pi(A \mid S)$ (a stochastic policy).*

stationary policy는 시간에 따라 바뀌지 않는 정책이다. $\pi : S \rightarrow A$(a deterministic policy) 또는 $\pi(A \mid S)$(a stochastic policy) 가 될 수 있는 모든 $\pi$에 대하여 모든 $t$에 대하여 $\pi_t=\pi$이다

***A non-stationary policy** is a policy that is not stationary. More precisely, $\pi_i$ may not be equal to $\pi_j$, for $i \neq j \ge 0$, where $i$ and $j$ are thus two different time steps.*

A nonstationary policy는 stationary하지 않은 policy이다. 더 정확히는 $0$이상의 서로 다른 시점의 $i,j$에서 정책은 서로 다를 수 있다.

*There are problems where a stationary optimal policy is guaranteed to exist.*

stationary optimal policy가 존재할 것이 보장되는 문제가 있다.

*For example, in the case of a stochastic (there is a probability density that models the dynamics of the environment, that is, the transition function and the reward function) and discrete-time Markov decision process (MDP) with finite numbers of states and actions, and bounded rewards, where the objective is the long-run average reward, a stationary optimal policy exists.*

 예를 들어, stochastic(환경 역학, 즉 전이 함수와 보상 함수를 모델링하는 확률밀도가 있다.)하고 이산 시간의 MDP에서 유한개의 state와 action, 그리고 제한된 보상 이때 목적이 장기 평균 보상인 경우 stationary optimal policy가 존재하는 것이 보장된다.

*The proof of this fact is in the book Markov Decision Processes: Discrete Stochastic Dynamic Programming (1994), by Martin L. Puterman, which apparently is not freely available on the web.*

 이 사실에 대한 증명은 Markov Decision Processes: Discrete Stochastic Dynamic Programming (1994), by Martin L. Puterman이라는 책에 있다, 웹에서는 자유로이 접근가능하지 않은 것으로 보인다.

---

`stationary bandit problems`에서는 보상값의 확률 분포가 시간이 지나도 변하지 않는다.
