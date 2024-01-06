---
layout: single
title: "(WIP)단단한 강화학습 ch13 : 정책 경사도 방법"
date: 2024-01-06 02:23:00
lastmod: 2024-01-06 02:23:00
categories: RL
tag: [RL, Sutton]
toc: true
toc_sticky: true
use_math: true
published: false
---

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha \widehat{\nabla J(\boldsymbol{\theta}_t)}, \tag{13.1}
$$

$$
\pi(a|s, \boldsymbol{\theta}) = \frac{e^{h(s,a,\boldsymbol{\theta})}}{\sum_b e^{h(s,b,\boldsymbol{\theta})}}\tag{13.2}
$$

$$
h(s, a, \boldsymbol{\theta}) = \boldsymbol{\theta}^\top \mathbf{x}(s, a) \tag{13.3}
$$

* $\mathbf{x}(s, a)$ : 상태 $s$에서 행동 a를 취할 떄 드러나는 특징 벡터

$$
J(\boldsymbol{\theta}) \doteq v_{\pi_{\boldsymbol{\theta}}}(s_0) \tag{13.4}
$$

$$
\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_a q_{\pi}(s, a) \nabla \pi(a|s, \boldsymbol{\theta})\tag{13.5}
$$

* 에피소딕 문제의 경우 : 비례 상수가 에피소드의 평균 길이
* 연속적 문제의 경우 : 비례 상수가 1이 됨으로써 관계식은 등식으로 바뀐다.

---

### 정책 경사도 정리의 증명(에피소딕 문제의 경우)

$$
\begin{align*}
\nabla v_{\pi}(s) &= \nabla \left[ \sum_a \pi(a|s)q_{\pi}(s, a) \right], \text{ for all } s \in \mathcal{S} && \text{(Exr. 3.18)}\\
&= \sum_a \left[ \nabla \pi(a|s)q_{\pi}(s, a) + \pi(a|s)\nabla q_{\pi}(s, a) \right] && \text{(product rule of calculus)} \\
&= \sum_a \left[ \nabla \pi(a|s)q_{\pi}(s, a) + \pi(a|s) \nabla \sum_{s',r} p(s',r|s,a) ( r + v_{\pi}(s') ) \right]  && \text{(Exr. 3.19 and Eq. 3.2)}\\
&= \sum_a \left[ \nabla \pi(a|s)q_{\pi}(s, a) + \pi(a|s) \sum_{s'} p(s'|s,a) \nabla v_{\pi}(s') \right] && \text{(Eq. 3.4)} \\
&= \sum_a \left[ \nabla \pi(a|s)q_{\pi}(s, a) + \pi(a|s) \sum_{s'} p(s'|s,a) \right. && \text{(unrolling)} \\
&\left. \qquad  \sum_{a'} [ \nabla \pi(a'|s')q_{\pi}(s', a') + \pi(a'|s') \sum_{s''} p(s''|s', a') \nabla v_{\pi}(s'')] \right] \\
&= \sum_{x \in \mathcal{S}} \sum_{k=0}^{\infty} \text{Pr}(s \rightarrow x, k, \pi) \sum_a \nabla \pi(a|x)q_{\pi}(x, a)
\end{align*}
$$

* $\text{Pr}(s \rightarrow x, k, \pi)$ : 정책 $\pi$ 하에서 상태 $s$에서 상태 $x$로 $k$단계만에 전이할 확률
* $s$에서 최종적으로 전이 가능한 모든 $x$를 고려한다, 근데 그 $x$는 첫번째만에 갈 수도 있고 두번째 만에 갈 수도 있으니 모든 경우를 고려한다. 그리고 그 $k$번째에 도착한 그 $x$에서 각 행동을 선택할 수 있는데 이때의 모든 행동 $a$를 고려한다.

$$
\begin{align*}
\nabla J(\boldsymbol{\theta}) &= \nabla v_{\pi}(s_0) \\
&= \sum_s \left( \sum_{k=0}^{\infty} \text{Pr}(s_0 \rightarrow s, k, \pi) \right) \sum_a \nabla \pi(a|s)q_{\pi}(s, a) \\
&= \sum_s \eta(s) \sum_a \nabla \pi(a|s)q_{\pi}(s, a) \\
&= \sum_{s'} \eta(s') \sum_{s} \frac{\eta(s)}{\sum_{s'} \eta(s')} \sum_a \nabla \pi(a|s)q_{\pi}(s, a) \\
&= \sum_{s'} \eta(s') \sum_{s} \mu(s) \sum_a \nabla \pi(a|s)q_{\pi}(s, a) && \text{(Eq. 9.3)} \\
&\propto \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_{\pi}(s, a) && \text{(Q.E.D.)}
\end{align*}
$$

* $\mu(s)$ : 상태에 대한 활성 정책 분포
* $\eta(s)$ : 에피소드당 상태 $s$를 마주치는 횟수의 기댓값

---

$$
\begin{align*}
  \nabla J(\boldsymbol{\theta}) & \propto \sum_s \mu(s) \sum_a q_{\pi}(s, a) \nabla \pi(a|s, \boldsymbol{\theta}) \\ 
  &= \mathbb{E}_{\pi} \left[ \sum_a q_{\pi}(S_t, a) \nabla \pi(a|S_t, \boldsymbol{\theta}) \right]. \tag{13.6}
\end{align*}
$$

$$
\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_t + \alpha \sum_a \hat{q}(S_t, a, \mathbf{w}) \nabla \pi(a|S_t, \boldsymbol{\theta}), \tag{13.7}
$$

* $\hat{q}$ : $q_\pi$에 대한 학습된 근삿값.

$$
\begin{align*}
\nabla J(\boldsymbol{\theta}) & \propto \mathbb{E}_{\pi} \left[\sum_a \pi(a|S_t, \boldsymbol{\theta}) q_{\pi}(S_t, a) \frac{\nabla \pi(a|S_t, \boldsymbol{\theta})}{\pi(a|S_t, \boldsymbol{\theta})} \right]\\
& = \mathbb{E}_{\pi} \left[ q_{\pi}(S_t, A_t) \frac{\nabla \pi(A_t|S_t, \boldsymbol{\theta})}{\pi(A_t|S_t, \boldsymbol{\theta})} \right] && (\text{replacing } a \text{ by the sample } A_t \sim \pi) \\
& = \mathbb{E}_{\pi} \left[ G_t \frac{\nabla \pi(A_t|S_t, \boldsymbol{\theta})}{\pi(A_t|S_t, \boldsymbol{\theta})} \right] && (\text{because } \mathbb{E}\left[ G_t \vert S_t, A_t \right] = q_\pi(S_t, A_t))
\end{align*}
$$

$$
\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_t + \alpha G_t \frac{\nabla \pi(A_t|S_t, \boldsymbol{\theta}_t)}{\pi(A_t|S_t, \boldsymbol{\theta}_t)} \tag{13.8}
$$

* $\frac{\nabla \pi (A_t \vert S_t, \boldsymbol{\theta}_t)}{\pi (A_t \vert S_t, \boldsymbol{\theta}_t)} = \nabla \ln \pi (A_t \vert S_t, \boldsymbol{\theta}_t)$


$$
\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_a \left( q_{\pi}(s, a) - b(s) \right) \nabla \pi(a|s, \boldsymbol{\theta}) \tag{13.10}
$$

$$
\sum_a b(s) \nabla \pi(a|s, \boldsymbol{\theta}) = b(s) \nabla \sum_a \pi(a|s, \boldsymbol{\theta}) = b(s) \nabla 1 = 0
$$

$$
\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_t + \alpha \left( G_t - b(S_t) \right) \frac{\nabla \pi(A_t|S_t, \boldsymbol{\theta}_t)}{\pi(A_t|S_t, \boldsymbol{\theta}_t)} \tag{13.11}
$$

$$
\begin{align*}
\boldsymbol{\theta}_{t+1} & \doteq \boldsymbol{\theta}_t + \alpha \left( G_{t:t+1} - \hat{v}(S_t, \mathbf{w}) \right) \frac{\nabla \pi(A_t|S_t, \boldsymbol{\theta}_t)}{\pi(A_t|S_t, \boldsymbol{\theta}_t)} \tag{13.12} \\
&= \boldsymbol{\theta}_t + \alpha \left( R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w}) \right) \frac{\nabla \pi(A_t|S_t, \boldsymbol{\theta}_t)}{\pi(A_t|S_t, \boldsymbol{\theta}_t)} \tag{13.13} \\
&= \boldsymbol{\theta}_t + \alpha \delta_t \frac{\nabla \pi(A_t|S_t, \boldsymbol{\theta}_t)}{\pi(A_t|S_t, \boldsymbol{\theta}_t)} \tag{13.14}
\end{align*}
$$

$$
\begin{align*}
J(\theta) \doteq r(\pi) & \doteq \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^{h} \mathbb{E}[R_t | S_0, A_{0:t-1} \sim \pi] \\
&= \lim_{t \to \infty} \mathbb{E}[R_t | S_0, A_{0:t-1} \sim \pi] \\
&= \sum_s \mu(s) \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) r,
\end{align*}\tag{13.10}
$$

* $\mu(s) \doteq \lim_{t \rightarrow \infty} \text{Pr}\{ S_t = s \vert A_{0:t} \sim \pi \}$

$$
\sum_s \mu(s) \sum_a \pi(a|s, \boldsymbol{\theta}) p(s'|s, a) = \mu(s'), \text{ for all } s' \in \mathcal{S}. \tag{13.16}
$$

$$
G_t \doteq R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + R_{t+3} - r(\pi) + \cdots .\tag{13.17}
$$

---

### (연속적 문제의 경우) 정책 경사도 정리의 증명

$$
\begin{align*}
\nabla v_{\pi}(s) &= \nabla \left[ \sum_a \pi(a|s)q_{\pi}(s, a) \right], \quad \text{ for all } s \in \mathcal{S} && \text{(Exr. 3.18)} \\
&= \sum_a \left[ \nabla \pi(a|s)q_{\pi}(s, a) + \pi(a|s)\nabla q_{\pi}(s, a) \right] && \text{(product rule of calculus)} \\
&= \sum_a \left[ \nabla \pi(a|s)q_{\pi}(s, a) + \pi(a|s) \sum_{s',r} p(s',r|s,a) \left( r - r(\boldsymbol{\theta}) + v_{\pi}(s') \right) \right] \\
&= \sum_a \left[ \nabla \pi(a|s)q_{\pi}(s, a) + \pi(a|s) [ -\nabla r(\boldsymbol{\theta}) + \sum_{s'} p(s'|s,a) \nabla v_{\pi}(s') ] \right].
\end{align*}
$$

$$
\begin{align*}
  \nabla r(\boldsymbol{\theta}) = \sum_a \left[ \nabla \pi(a|s)q_{\pi}(s, a) + \pi(a|s) \sum_{s'} p(s'|s, a) \nabla v_{\pi}(s') \right] - \nabla v_{\pi}(s).
\end{align*}
$$

$$
\begin{align*}
\nabla J(\boldsymbol{\theta}) & = \sum_s \mu(s) \left( \sum_a \left[ \nabla \pi(a|s)q_{\pi}(s, a) + \pi(a|s) \sum_{s'} p(s'|s, a) \nabla v_{\pi}(s') \right] - \nabla v_{\pi}(s) \right) \\
& = \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_{\pi}(s, a) \\ & \qquad + \sum_s \mu(s) \sum_a \pi(a|s) \sum_{s'} p(s'|s, a) \nabla v_{\pi}(s') - \sum_s \mu(s) \nabla v_{\pi}(s) \\
& = \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_{\pi}(s, a) \\
& \qquad + \sum_{s'} \underbrace{\sum_s \mu(s) \sum_a \pi(a|s)p(s'|s, a)}_{\mu(s') \ (13.16)} \nabla v_{\pi}(s') - \sum_s \mu(s) \nabla v_{\pi}(s) \\
& = \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_{\pi}(s, a) + \sum_{s'} \mu(s') \nabla v_{\pi}(s') - \sum_s \mu(s) \nabla v_{\pi}(s) \\
& = \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_{\pi}(s, a). \qquad \text{Q.E.D.}
\end{align*}
$$

---

$$
p(x) \doteq \frac{1}{\sigma\sqrt{2\pi}} \exp \left( -\frac{(x - \mu)^2}{2\sigma^2} \right)\tag{13.18}
$$

$$
\pi(a|s, \boldsymbol{\theta}) \doteq \frac{1}{\sigma(s, \boldsymbol{\theta})\sqrt{2\pi}} \exp \left( -\frac{(a - \mu(s, \boldsymbol{\theta}))^2}{2\sigma(s, \boldsymbol{\theta})^2} \right), \tag{13.19}
$$

* $\mu$ : $\mathcal{S} \times \mathbb{R}^{d'} \rightarrow \mathbb{R}$
* $\sigma$ : $\mathcal{S} \times \mathbb{R}^{d'} \rightarrow \mathbb{R}^{+}$

$$
\mu(s, \boldsymbol{\theta}) \doteq \boldsymbol{\theta}_{\mu}^\top \mathbf{x}_{\mu}(s) \quad \text{and} \quad \sigma(s, \boldsymbol{\theta}) \doteq \exp \left(\boldsymbol{\theta}_{\sigma}^\top \mathbf{x}_{\sigma}(s) \right), \tag{13.20}
$$
