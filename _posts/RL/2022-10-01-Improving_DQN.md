---
title: "향상된 DQN"
date: 2022-10-01 19:13:10
lastmod : 2022-10-01 19:13:13
categories: RL
tag: [RL, DQN]
toc: true
toc_sticky: true
use_math: true
---

# Target Networks

원래의 `DQN` 알고리즘에서 $Q^{\pi}_{\text{tar}}$가 $\hat{Q^\pi}(s,a)$에 따라 결정되기 때문에 지속적으로 값이 변한다는 문제를 해결하기 위해 만들어졌다.

훈련 도중에 $\hat{Q^\pi}(s,a)=Q^{\pi_\theta}(s,a)$ 와 $$Q^{\pi}_\text{tar}$$ 의 차이를 최소화하기 위해 $$Q$$ 네트워크 파라미터 $\theta$를 조정하는데, $Q^{\pi}_\text{tar}$의 값이 훈련 단계마다 변하는 경우 이러한 조정이 어려워진다.

훈련 단계가 바뀔 때 $$Q^{\pi}_\text{tar}$$의 변화를 최소화 하기 위해 목표 네트워크를 사용한다. 목표 네트워크는 파라미터 $$\varphi$$를 갖는 네트워크로 $$Q$$ 네트워크 $$Q^{\pi_\theta}(s,a)$$의 지연된 버전이다.

아래 $(5.2)$식에서 볼 수 있듯이 목표네트워크 $Q^{\pi_\varphi}(s,a)$는 $Q^{\pi}_\text{tar}$를 계산하기 위해 사용된다.

$$Q^{\pi_\varphi}_{\text{tar}}(s,a)=r+\gamma \underset{a'}{\max}Q^{\pi_\varphi}(s',a') \tag{5.2}$$

$\varphi$는 $\theta$의 현재 값으로 주기적으로 업데이트된다. 이것을 치환 업데이트(replacement update)라고 부른다. $\varphi$의 업데이트 주기는 문제마다 다르며 간단한 문제일 수록 업데이트 주기가 작아도 충분하다.

$Q^{\pi_\theta}_{\text{tar}}(s,a)$가 계산될 때마다, 파라미터 $$\theta$$로 표현되는 $Q$ 함수는 조금 달라질 것이기 때문에 $$Q^{\pi_\theta}_{\text{tar}}(s,a)$$는 동일한 $$(s,a)$$에 대해 다른 값을 갖게 될 것이다.

이러한 '움직이는 목표'는 네트워크가 어떤 값을 도출해야 하는지를 모호하게 만들기 떄문에 훈련이 불안정해진다.

목표 네트워크를 도입하면 $\varphi$를 $\theta$로 업데이트 하는 사이에 $\varphi$가 고정되기 때문에 $\varphi$로 표현되는 $Q$함수가 변하지 않는다. 이렇게 함으로써 문제를 지도 회귀(supervised regression) 문제로 전환할 수 있다.

![fdrl_algorithm_5_1_1](../../assets/images/rl/fdrl_algorithm_5_1_1.png){: width="80%" height="80%" class="align-center"}

이전 [DQN 포스트의 [볼츠만 정책을 적용한 DQN]](https://helpingstar.github.io/rl/DQN/#%EB%B3%BC%EC%B8%A0%EB%A7%8C-%EC%A0%95%EC%B1%85%EC%9D%84-%EC%A0%81%EC%9A%A9%ED%95%9C-dqn)의 알고리즘과 다른 점은 다음과 같다.

* `(7)` : 목표 업데이트 빈도수 $F$ 추가
* `(9)` : 추가적인 네트워크를 목표 네트워크로 초기화하고 $\varphi$를 $\theta$로 설정한다.
* `(17)` : 목표 네트워크 $Q^{\pi_\varphi}_{\text{tar}}(s,a)$를 이용해서 $y_i$를 계산한다.
* `(26~29)` : 목표 네트워크는 주기적으로 업데이트된다.

목표 네트워크 파라미터 $\varphi$를 업데이트 하는 것은 두 가지 방법이 있다

$$\text{Replacement update : } \varphi \leftarrow \theta \tag{5.3}$$
$$\text{Polyak update : } \varphi \leftarrow \beta\varphi + (1-\beta)\theta$$

폴리악 업데이트(Polyak update)는 $\varphi$를 $\varphi$와 $\theta$의 가중평균으로 설정하는데 이는 업데이트를 부드럽게 한다. $\varphi$는 시간 단계마다 변하지만 $\theta$보다는 천천히 변하며 $\beta$로 $\varphi$의 변화속도를 조절한다. $\beta$가 클수록 $\varphi$는 천천히 변화한다.

**치환 업데이트(Replacement update)**
* $\varphi$가 수많은 단계동안 유지된다. 이는 움직이는 `target`을 제거하는 효과가 생긴다.
* $\varphi$와 $\theta$의 동역학적 지연를 발생시킨다. 그리고 이러한 지연은 $\varphi$가 마지막으로 업데이트된 이후 경과한 시간 단계의 개수에 영향을 받는다.

**폴리악 업데이트**
* $\varphi$는 훈련의 반복 과정 속에서 변화하지만 $\theta$보다는 덜 점진적으로 변화한다.
* $\varphi$와 $\theta$를 섞어놓은 것이 변하지 않기 때문에 시간 단계의 개수에 영향을 받는 동역학적 지연이 없다.

목표 네트워크의 한 가지 단점은 $Q^{\pi}_\text{tar}(s,a)$가 이전의 목표 네트워크로부터 생성되기 때문에 훈련 속도가 저하될 수 있다는 점이다. $\varphi$와 $\theta$가 너무 비슷한 값을 갖는다면 훈련 과정은 불안정해지겠지만 $\varphi$가 너무 천천히 변화한다면 훈련 과정은 불필요하게 느려질 것이다. $\varphi$의 변화 속도를 조절하는 하이퍼 파라미터(업데이트 빈도수 또는 $\beta$)는 안정성과 훈련속도 사이의 적절한 균형을 찾도록 조절되어야 한다.

<!--
* $Q^{\pi}_\text{tar}$
* $\hat{Q^\pi}(s,a)$
* $Q^{\pi_\theta}(s,a)$
* $Q^{\pi_\varphi}(s,a)$
* $Q^{\pi_\varphi}_{\text{tar}}(s,a)$
* $Q^{\pi_\theta}_{\text{tar}}(s,a)$
-->
