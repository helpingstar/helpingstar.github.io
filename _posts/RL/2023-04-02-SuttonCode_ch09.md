---
layout: single
title: "단단한 강화학습 코드 정리, chap9"
date: 2023-04-02 14:13:45
lastmod : 2023-04-02 14:13:45
categories: RL
tag: [Sutton, 단단한 강화학습, RL]
toc: true
toc_sticky: true
use_math: true
---

[ShangtongZhang github](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/tree/master/chapter09)

[단단한 강화학습](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791190665179&orderClick=LAG&Kc=) 책의 코드를 공부하기 위해 쓰여진 글이다.

# `step(state, action)`

```python
# take an @action at @state, return new state and reward for this transition
def step(state, action):
    step = np.random.randint(1, STEP_RANGE + 1)
    step *= action
    state += step
    state = max(min(state, N_STATES + 1), 0)
    if state == 0:
        reward = -1
    elif state == N_STATES + 1:
        reward = 1
    else:
        reward = 0
    return state, reward
```
* **(3)** : $[1, \text{STEP\_RANGE}]$ 범위에서 한 정수를 무작위로 추출하여 `step`에 대입한다.
* **(4)** : 해당 `step`에 `action`을 곱한다 (`action`은 (-1 / 1)로 (왼쪽 / 오른쪽)을 의미한다.)
* **(5)** : 현재 `state`에 `step`을 더하여 `state`를 옮긴다.
* **(6)** : `state`가 0보다 작을 경우 0으로 `N_STATES+1`보다 클 경우 `N_STATES+1`로 클리핑한다.
* **(7~12)** : `state`가 0(왼쪽 끝)일 경우 -1, `N_STATES`+1 일 경우 1, 그 외는 0를 `reward`로 받는다.
* **(13)** : `state`(새로운 상태), `reward를` 반환한다.

# `get_action()`
```python
# get an action, following random policy
def get_action():
    if np.random.binomial(1, 0.5) == 1:
        return 1
    return -1
```
* **(3)** : 

`numpy.random.binomial(n, p, size=None)` : Draw samples from a binomial distribution.

p의 당첨확률을 가진 복권을 n개 사서 몇개가 당첨될지 테스트를 size번 하는 것이다.

이항분포 확률에 따라 0에서 n까지의 숫자중 하나를 출력한다. 0부터 n까지의 숫자 중 어떤 숫자 x(p에 몇번 해당되었는지)가 산출될 확률은 아래와 같다.

$$P(N)=\binom{n}{x}p^x(1-p)^{n-x}$$

코드에서는 `np.random.binomial(1, 0.5)`으로 되어 있는데 그러면 반환값이 1이 나올 확률이 `0.5`이 된다는 뜻이다. 그러므로 해당 조건문은 `0.5`의 확률로 `True`를 반환한다.


# `ValueFunction`

```python
# a wrapper class for aggregation value function
class ValueFunction:
    # @num_of_groups: # of aggregations
    def __init__(self, num_of_groups):
        self.num_of_groups = num_of_groups
        self.group_size = N_STATES // num_of_groups

        # thetas
        self.params = np.zeros(num_of_groups)

    # get the value of @state
    def value(self, state):
        if state in END_STATES:
            return 0
        group_index = (state - 1) // self.group_size
        return self.params[group_index]

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    def update(self, delta, state):
        group_index = (state - 1) // self.group_size
        self.params[group_index] += delta
```
* **(5)** : `self.num_of_groups` : 결집의 개수(몇 개로 결집할 것인지)
* **(6)** : `self.group_size` : 한 결집에 포함된 상태의 개수 (`N_STATES` == 1000)
* 

# `gradient_monte_carlo`
```python
# gradient Monte Carlo algorithm
# @value_function: an instance of class ValueFunction
# @alpha: step size
# @distribution: array to store the distribution statistics
def gradient_monte_carlo(value_function, alpha, distribution=None):
    state = START_STATE
    trajectory = [state]

    # We assume gamma = 1, so return is just the same as the latest reward
    reward = 0.0
    while state not in END_STATES:
        action = get_action()
        next_state, reward = step(state, action)
        trajectory.append(next_state)
        state = next_state

    # Gradient update for each state in this trajectory
    for state in trajectory[:-1]:
        delta = alpha * (reward - value_function.value(state))
        value_function.update(delta, state)
        if distribution is not None:
            distribution[state] += 1
```
* **(6)** : 현재 상태 `state`를 `START_STATE` == 500 으로 둔다
* **(7)** : trajectory를 추적할 리스트의 초깃값으로 `state`를 넣는다.
* **(9)** : termination에서 1 혹은 -1의 보상을 얻고 이외의 상태에서는 0의 보상을 얻으므로 $\gamma=1$이면 return이 마지막의 보상과 같다.

$$G_t \doteq R_{t+1}+\gamma R_{t+2} + \gamma^{2}R_{t+3} + \cdots = \sum^{\infty}_{k=0} \gamma^{k} R_{t+k+1}$$

* **(11)** : state가 `END_STATES = [0, N_STATES + 1]`에 포함되지 않을 때까지 (양 끝에 도달하기 전까지) 반복문을 실행한다.
* **(12)** : 행동 둘 중 하나를 0.5의 확률로 정한다.
* **(13)** : `step` 함수에 현재 상태(state)와 행동(action)을 인수로 집어넣어 다음 상태(next_state)와 보상(reward)를 받는다.
* **(14)** : **(7)**에서 정의한 `trajectory`에 **(13)**에서 얻은 새로운 상태(next_state)를 append한다.
* **(15)** : 새로운 상태를 state에 저장한다.


# figure 9.1

```python
# Figure 9.1, gradient Monte Carlo algorithm
def figure_9_1(true_value):
    episodes = int(1e5)
    alpha = 2e-5

    # we have 10 aggregations in this example, each has 100 states
    value_function = ValueFunction(10)
    distribution = np.zeros(N_STATES + 2)
    for ep in tqdm(range(episodes)):
        gradient_monte_carlo(value_function, alpha, distribution)

    distribution /= np.sum(distribution)
    state_values = [value_function.value(i) for i in STATES]
```
* **(4)** : `alpha` : step size
* **(7)** : 
* **(8)** :
* **(9~10)** : `episodes` 개수만큼 `gradient_monte_carlo` 를 수행한다.