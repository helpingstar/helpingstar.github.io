---
layout: single
title: "딥러닝 옵티마이저, Optimizer"
date: 2022-08-10 05:03:56
lastmod : 2022-08-10 05:03:59
categories: optimization
tag: [optimizer, momentum, nesterov, AdaGrad, RMSProp, Adam, Nadam]
toc: true
toc_sticky: true
use_math: true
---

# 경사 하강법

경사 하강법은 가중치에 대한 비용 함수 $J(\theta)$의 그레이디언트 ($\nabla_{\theta}J(\theta)$)에 학습률 $\eta$를 곱한 것을 바로 차감하여 $\theta$를 갱신한다. 공식은 $\theta \leftarrow \theta-\eta \nabla_{\theta}J(\theta)$ 이다. 이 식은 이전 그레이디언트가 얼마였는지 고려하지 않는다. 국부적으로 그레이디언트가 아주 작으면 매우 느려질 것이다.

# 모멘텀 최적화, Momentum

모멘텀 최적화는 이전 그레이디언트를 고려한다. 매 반복에서 현재 그레이디언트를 (학습률 $\eta$를 곱한 후) **모멘텀 벡터** $\bold{m}$에 더하고 이 값을 빼는 방식으로 가중치를 갱신한다.

다시 말해 그레이디언트를 속도가 아니라 가속도로 사용한다. 

*[Momentum Algorithm]*

1. $\bold{m} \leftarrow \beta\bold{m}-\eta \nabla_{\theta}J(\theta)$
2. $\theta \leftarrow \theta+\bold{m}$

* $\beta$ : 모멘텀(마찰저항), 모멘텀이 너무 커지는 것을 막기 위한 파라미터이다. 0(높은 마찰저항), 1(마찰저항 없음) 사이로 설정되어야 한다. 일반적으로 0.9

그레이디언트가 일정하다면 종단속도(즉, 가중치를 갱신하는 최대 크기)는 학습률 $\eta$를 곱한 그레이디언트에 $\frac{1}{1-\beta}$를 곱한 것과 같다. 

종단속도는 등속도 운동이므로

$\bold{m} = \beta\bold{m}-\eta \nabla_{\theta}J(\theta)$

$\bold{m}(1-\beta) = -\eta\nabla_{\theta}J(\theta)$

$\bold{m} = -\frac{1}{1-\beta}\eta\nabla_{\theta}J(\theta)$


$$\begin{align} \bold{m} = \beta\bold{m}-\eta \nabla_{\theta}J(\theta) \\ \bold{m}(1-\beta) = -\eta\nabla_{\theta}J(\theta) \end{align} \\ \bold{m} = -\frac{1}{1-\beta}\eta\nabla_{\theta}J(\theta)$$

$\nabla_{\theta}J(\theta)$