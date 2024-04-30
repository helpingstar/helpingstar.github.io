---
layout: single
title: "(WIP)Policy Gradient"
date: 2024-04-10 17:00:00
lastmod: 2024-04-10 17:00:00
categories: RL
tag: [RL, Policy Gradient]
toc: true
toc_sticky: true
use_math: true
published: true
---

## Policy Gradient

* $p\_\theta(\tau)$ : 정책 $\pi\_\theta$로 생성되는 궤적의 확률밀도함수

확률의 연쇄법칙을 적용해 $p\_\theta(\tau)$를 전개하면 다음과 같다.

$$
\begin{align*}
p_\theta(\tau) &= p_\theta(\mathbf{s}_1, \mathbf{a}_1, \ldots, \mathbf{s}_T, \mathbf{a}_T) \\
&= p(\mathbf{s}_1)p_\theta(\mathbf{a}_1, \mathbf{s}_2, \ldots, \mathbf{s}_T, \mathbf{a}_T \mid \mathbf{s}_1) \\
&= p(\mathbf{s}_1)p_\theta(\mathbf{a}_1 \vert \mathbf{s}_1)p_\theta(\mathbf{s}_2, \mathbf{a}_2, \ldots, \mathbf{s}_T, \mathbf{a}_T \vert \mathbf{s}_1, \mathbf{a}_1) \\
&= p(\mathbf{s}_1)p_\theta(\mathbf{a}_1 \vert \mathbf{s}_1)p(\mathbf{s}_2 \vert \mathbf{s}_1, \mathbf{a}_1) p_\theta(\mathbf{a}_2, \mathbf{s}_3, \ldots, \mathbf{s}_T, \mathbf{a}_T \vert \mathbf{s}_1, \mathbf{a}_1, \mathbf{s}_2)
\end{align*}
$$

* $p(\mathbf{s}\_2 \vert \mathbf{s}\_1, \mathbf{a}\_1)$ : 환경 모델로 정책과 무관하기에 아래 첨자 없이 표기

마르코프 시퀀스 가정에 의해 아래를 만족한다.

$$
\begin{align*}
& p_\theta(\mathbf{a}_2 \vert \mathbf{s}_1, \mathbf{a}_1, \mathbf{s}_2) = \pi_\theta(\mathbf{a}_2 \vert \mathbf{s}_2) \\
& p(\mathbf{s}_3 \vert \mathbf{s}_1, \mathbf{a}_1, \mathbf{s}_2, \mathbf{a}_2) = p(\mathbf{s}_3 \vert \mathbf{s}_2, \mathbf{a}_2)
\end{align*}
$$

그러므로 아래와 같은 식을 얻을 수 있다.

$$
p_\theta(\tau) = p_\theta(\mathbf{s}_1, \mathbf{a}_1, \ldots, \mathbf{s}_T, \mathbf{a}_T) = p(\mathbf{s}_1) \prod_{t=1}^T \pi_\theta(\mathbf{a}_t \vert \mathbf{s}_t) p(\mathbf{s}_{t+1} \vert \mathbf{s}_t, \mathbf{a}_t)
$$

우리의 목적은 목적함수 $J(\theta)=E\_{\tau \sim p\_{\theta}(\tau)} \left[ \sum\_t r(\mathbf{s}\_t, \mathbf{a}\_t) \right]$를 최대화하는 정책 파라미터 $\theta$를 계산하는 것이다.

<!-- TODO infinite, finite 차이 -->

$$
\theta^* = \arg\max_{\theta} E_{\tau \sim p_{\theta}(\tau)} \left[ \sum_t r(\mathbf{s}_t, \mathbf{a}_t) \right]
$$

* infinite horizon Case
$$
\theta^* = \arg\max_{\theta} E_{(\mathbf{s},\mathbf{a})\sim p_{\theta}(\mathbf{s},\mathbf{a})} \left[ r(\mathbf{s}, \mathbf{a}) \right]
$$

* finite horizon case
$$
\theta^* = \arg\max_{\theta} \sum_{t=1}^{T} E_{(\mathbf{s}_t,\mathbf{a}_t) \sim p_{\theta}(\mathbf{s}_t,\mathbf{a}_t)} \left[ r(\mathbf{s}_t, \mathbf{a}_t) \right]
$$

기댓값은 실제로 계산할 수 없으므로 샘플을 이용해 추정한다. 기댓값 $\mathbb{E}\_{\tau \sim p\_\theta(\tau)}[\cdot]$은 에피소드를 $\pi\_\theta$ 정책을 이용해 $N$개만큼 생성해 에피소드 평균을 이용해서 근사적으로 계산한다.

$$
\mathbb{E}_{\tau \sim p_\theta(\tau)}[\cdot] \approx \frac{1}{N} \sum_{i}^M [\cdot]
$$

위 공식을 이용하여 목적함수를 다음과 같이 근사적으로 추정한다.

$$
J(\theta) = E_{\tau\sim p_{\theta}(\tau)} \left[ \sum_t r(\mathbf{s}_t, \mathbf{a}_t) \right] \approx \frac{1}{N} \sum_i \sum_t r(\mathbf{s}_{i,t}, \mathbf{a}_{i,t})
$$

다시 강화학습의 목적을 보자.

우리의 목적은 목적함수 $J(\theta)=E\_{\tau \sim p\_{\theta}(\tau)} \left[ \sum\_t r(\mathbf{s}\_t, \mathbf{a}\_t) \right]$를 최대화하는 정책 파라미터 $\theta$를 계산하는 것이었다.

$$
\theta^* = \arg\max_{\theta} E_{\tau \sim p_{\theta}(\tau)} \left[ \sum_t r(\mathbf{s}_t, \mathbf{a}_t) \right]
$$

우리는 아래 목적함수를 최대화하는 방향을 찾기 위해 아래 함수를 미분해야 한다.

$$
J(\theta) = E_{\tau \sim p_{\theta}(\tau)} \left[ \sum_t r(\mathbf{s}_t, \mathbf{a}_t) \right]
$$

식의 편의성을 위해 다음과 같이 표기한다.

$$
\sum_{t=1}^T r(\mathbf{s}_t, \mathbf{a}_t) = r(\tau)
$$

우리의 목적함수를 적분으로 표현하면 다음과 같다.

$$
J(\theta) = E_{\tau \sim p_{\theta}(\tau)} \left[ \sum_{t=1}^T r(\mathbf{s}_t, \mathbf{a}_t) \right] = E_{\tau \sim p_{\theta}(\tau)} \left[ r(\tau) \right] = \int p_{\theta}(\tau) r(\tau) d\tau
$$

위 식을 $\theta$에 대해 미분해보자

$$
\nabla_{\theta} J(\theta) = \int \nabla_{\theta} p_{\theta}(\tau) r(\tau) d\tau
$$

여기서 중요한 트릭이 등장한다.

$$
\nabla_\theta p_\theta(\tau) = p_{\theta}(\tau) \frac{\nabla_{\theta} p_{\theta}(\tau)}{p_{\theta}(\tau)} = p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau)
$$

에서 $\nabla\_\theta p\_\theta(\tau)$를 위 수식에 대입하면

$$
\nabla_{\theta} J(\theta) = \int \nabla_{\theta} p_{\theta}(\tau) r(\tau) d\tau = \int p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) r(\tau) d\tau
$$

식이 나오고 마지막 식을 기댓값으로 표현하면

$$
\nabla_{\theta} J(\theta) = \int p_{\theta}(\tau) \left[ \nabla_{\theta} \log p_{\theta}(\tau) r(\tau) \right] d\tau = E_{\tau\sim p_{\theta}(\tau)} \left[ \nabla_{\theta} \log p_{\theta}(\tau) r(\tau) \right]
$$

와 같이 표현된다. 이때 $\nabla\_\theta \log p\_{\theta}(\tau)$를 자세히 살펴보자 우리는 아래와 같이 $p\_{\theta}(\tau)$를 표현했다.

$$
p_\theta(\tau) = p_\theta(\mathbf{s}_1, \mathbf{a}_1, \ldots, \mathbf{s}_T, \mathbf{a}_T) = p(\mathbf{s}_1) \prod_{t=1}^T \pi_\theta(\mathbf{a}_t \vert \mathbf{s}_t) p(\mathbf{s}_{t+1} \vert \mathbf{s}_t, \mathbf{a}_t)
$$

$\log p\_{\theta}(\tau)$를 구하기 위해 위 식에 $\log$를 씌우면 식이 아래와 같아진다.

$$
\log p_{\theta}(\tau) = \log p(\mathbf{s}_1) + \sum_{t=1}^{T} \log \pi_{\theta}(\mathbf{a}_t \vert \mathbf{s}_t) + \log p(\mathbf{s}_{t+1} \vert \mathbf{s}_t, \mathbf{a}_t)
$$

마지막으로 최종 목표인 $\nabla\_\theta \log p\_{\theta}(\tau)$를 구하기 위해 $\nabla\_\theta$를 씌우면

$$
\nabla_\theta \log p_{\theta}(\tau) = \nabla_\theta \left[ \log p(\mathbf{s}_1) + \sum_{t=1}^{T} \log \pi_{\theta}(\mathbf{a}_t \vert \mathbf{s}_t) + \log p(\mathbf{s}_{t+1} \vert \mathbf{s}_t, \mathbf{a}_t) \right]
$$

인데 가운데 항만 $\theta$의 함수이므로 $\log p(\mathbf{s}\_1)$, $\log p(\mathbf{s}\_{t+1} \vert \mathbf{s}\_t, \mathbf{a}\_t)$를 제거하면

$$
\require{cancel}
\begin{align*}
\nabla_\theta \log p_{\theta}(\tau) &= \nabla_\theta \left[ \cancel{\log p(\mathbf{s}_1)} + \sum_{t=1}^{T} \log \pi_{\theta}(\mathbf{a}_t \vert \mathbf{s}_t) + \cancel{\log p(\mathbf{s}_{t+1} \vert \mathbf{s}_t, \mathbf{a}_t)} \right] \\
&= \nabla_\theta \left[\sum_{t=1}^{T} \log \pi_{\theta}(\mathbf{a}_t \vert \mathbf{s}_t) \right] \\
&= \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(\mathbf{a}_t \vert \mathbf{s}_t)
\end{align*}
$$

미분의 선형성에 의해 함수의 합은 미분의 각 함수의 미분의 합과 같으므로 마지막과 같이 표기한다.

이제 구한 식을 원래 목표였던 $\nabla\_{\theta} J(\theta)$ 식에 대입하자. 그리고 편의를 위해 잠시 바꿨던 $r(\tau)$도 돌려놓자

$$
\begin{align*}
\nabla_{\theta} J(\theta) &= \int \nabla_{\theta} p_{\theta}(\tau) r(\tau) d\tau = \int p_{\theta}(\tau) \left(\nabla_{\theta} \log p_{\theta}(\tau) \right) r(\tau) d\tau \\
&= \int p_{\theta}(\tau) \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(\mathbf{a}_t \vert \mathbf{s}_t) \right) r(\tau) d\tau \\
&= \int p_{\theta}(\tau) \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(\mathbf{a}_t \vert \mathbf{s}_t) \right) \left( \sum_{t=1}^T r(\mathbf{s}_t, \mathbf{a}_t) \right) d\tau \\
&= E_{\tau\sim p_{\theta}(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(\mathbf{a}_t \vert \mathbf{s}_t) \right) \left( \sum_{t=1}^T r(\mathbf{s}_t, \mathbf{a}_t) \right) \right]
\end{align*}
$$

기댓값을 추정하기 위해 샘플로 추정했던 아래 식을 떠올려보자

$$
J(\theta) = E_{\tau\sim p_{\theta}(\tau)} \left[ \sum_t r(\mathbf{s}_t, \mathbf{a}_t) \right] \approx \frac{1}{N} \sum_i \sum_t r(\mathbf{s}_{i,t}, \mathbf{a}_{i,t})
$$

같은 원리로 적용하면 다음과 같다.

$$
\begin{align*}
\nabla_{\theta} J(\theta) &= E_{\tau\sim p_{\theta}(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(\mathbf{a}_t \vert \mathbf{s}_t) \right) \left( \sum_{t=1}^T r(\mathbf{s}_t, \mathbf{a}_t) \right) \right] \\
&\approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(\mathbf{a}_{i,t} \vert \mathbf{s}_{i,t}) \right) \left( \sum_{t=1}^T r(\mathbf{s}_{i,t}, \mathbf{a}_{i,t}) \right)
\end{align*}
$$

<!-- TODO Continuous Action space -->

## Reduce Variance

### Casuality

목적함수의 그래디언트 식을 보자

$$
\begin{align*}
\nabla_{\theta} J(\theta) &= E_{\tau\sim p_{\theta}(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(\mathbf{a}_t \vert \mathbf{s}_t) \right) \left( \sum_{t=1}^T r(\mathbf{s}_t, \mathbf{a}_t) \right) \right] \\
&\approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(\mathbf{a}_{i,t} \vert \mathbf{s}_{i,t}) \right) \left( \sum_{t=1}^T r(\mathbf{s}_{i,t}, \mathbf{a}_{i,t}) \right)
\end{align*}
$$

$\left( \sum\_{t=1}^T r(\mathbf{s}\_t, \mathbf{a}\_t) \right)$ 은 $(t=1)$부터 에피소드가 종료$(t=T)$될 때까지 받을 수 있는 전체 보상의 합이다. 하지만 시간 $t'$에서의 정책은 시간 $t<t'$ 인 시간 $t$에서의 보상에 영향을 끼치지 않는다. 이러한 인과성(casuality)를 고려하여 위 식을 아래와 같이 수정할 수 있다.

$$
\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(\mathbf{a}_{i,t} \vert \mathbf{s}_{i,t}) \left( \sum_{t'=t}^T r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'}) \right)
$$

오른쪽 식 $\left( \sum\_{t'=t}^T r(\mathbf{s}\_{i,t'}, \mathbf{a}\_{i,t'}) \right)$은  reward-to-go 라고 하며 지금부터 에피소드 종료까지 얻을 보상을 의미한다.

이와 같이 현재 이전의 보상을 제외하면서 곱해지는 보상의 크기를 줄여 분산을 낮춘다.

### Baseline

우리는 목적함수의 그래디언트를 다음과 같이 구했다. (전개의 편의성을 위해 casuality를 적용하지 않는다.)

$$
\nabla_{\theta} J(\theta) = E_{\tau\sim p_{\theta}(\tau)} \left[ \nabla_{\theta} \log p_{\theta}(\tau) r(\tau) \right]
$$

식이 바뀌었는데 아래식이랑 같은 의미이다. 간단히 다시 말하면 위 식에서 $p\_{\theta}(\tau)$에 로그를 씌워서 순차적으로 곱해져있던 trajectory를 더하기로 바꾸고 미분으로 환경모델 확률식을 제거하여 아래와 같은 식이 나온다. 전개의 편의를 위해 다시 $r(\tau)=\sum\_{t=1}^T r(\mathbf{s}\_{t}, \mathbf{a}\_{t})$를 적용한다. 

$$
\nabla_{\theta} J(\theta) = E_{\tau\sim p_{\theta}(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_{\theta}(\mathbf{a}_t \vert \mathbf{s}_t) \right) \left( \sum_{t=1}^T r(\mathbf{s}_t, \mathbf{a}_t) \right) \right]
$$

근데 여기서 아래와 같이 전체 보상의 합을 계산하는 부분에 아래와 같이 상수를 빼도 될까?

$$
\nabla_{\theta} J(\theta) = E_{\tau\sim p_{\theta}(\tau)} \left[ \nabla_{\theta} \log p_{\theta}(\tau) \left( r(\tau) - b \right) \right]
$$

$b$만 따로 빼서 전개해보자

$$
E_{\tau\sim p_{\theta}} \left[ \nabla_{\theta} \log p_{\theta}(\tau) b \right] = \int p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) b \  d\tau
$$

우리는 아래와 같은 트릭을 사용했었다.

$$
\nabla_\theta p_\theta(\tau) = p_{\theta}(\tau) \frac{\nabla_{\theta} p_{\theta}(\tau)}{p_{\theta}(\tau)} = p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau)
$$

이번엔 거꾸로 $p\_{\theta}(\tau) \nabla\_{\theta} \log p\_{\theta}(\tau)$ 에 $\nabla\_\theta p\_\theta(\tau)$를 대입하면 아래와 같은 식이 나온다.

$$
\begin{align*}
E_{\tau\sim p_{\theta}} \left[ \nabla_{\theta} \log p_{\theta}(\tau) b \right] &= \int p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) b \  d\tau \\
&=\int \nabla_{\theta} p_{\theta}(\tau) b \  d\tau \\
&= b \nabla_{\theta} \int p_{\theta}(\tau) d\tau \\
&= b \nabla_{\theta} 1 = 0
\end{align*}
$$

그러므로 목적함수 그래디언트식의 $r(\tau)$에서 베이스라인을 빼도 기댓값은 변하지 않는다(unbiased). 