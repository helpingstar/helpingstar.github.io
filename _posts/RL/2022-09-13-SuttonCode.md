---
layout: single
title: "단단한 강화학습 코드 정리"
date: 2022-09-13 14:17:10
lastmod : 2022-09-13 14:17:10
categories: RL
tag: [Sutton, 단단한 강화학습]
toc: true
toc_sticky: true
use_math: true
---

https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter02/ten_armed_testbed.py

[단단한 강화학습](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791190665179&orderClick=LAG&Kc=) 책의 코드를 공부하기 위해 쓰여진 글이다.

# **`figure 2.1`**

![figure_2_1](../../assets/images/RL/sutton_figure_code/figure_2_1.png){: width="80%" height="80%" class="align-center"}

```python
def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('./images/figure_2_1.png')
    plt.close()
```
**`numpy.random.randn`: Return a sample (or samples) from the “standard normal” distribution.**


`(1)`

`dataset=np.random.randn(200, 10) + np.random.randn(10)` : (200, 10) 크기의 표준 정규분포에 각 행마다 10개의 요소가 있을텐데 뒤의 (1, 10) 크기의 표준 정규분포를 각각 더한다, 뒤의 `+np.random.randn(10)`은 평행이동 역할을 한다. 해당 코드를 빼면 아래와 같은 모양이 나온다.

![figure_2_1_1](../../assets/images/RL/sutton_figure_code/figure_2_1_1.png){: width="50%" height="50%" class="align-center"}

본 그림과 비교해보면 평균이 0에 더 가까운 것을 볼 수 있다.

**`violinplot` : Make a violin plot for each column of dataset or each vector in sequence dataset.**

각각의 column별로 값들을 plot한다

# **`Bandit`**

```python
class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward
        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial
        # # of chosen times for each action
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.time = 0

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward
```

# **`figure 2.2`**

![figure_2_2](../../assets/images/RL/sutton_figure_code/figure_2_2.png){: width="80%" height="80%" class="align-center"}

```python
def figure_2_2(runs=2000, time=1000):
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('./images/figure_2_2.png')
    plt.close()
```

`(3)` : `sample_averages=True`인 `Bandit` 클래스를 `epsilon` 별로 생성한다.

`(4)` : 