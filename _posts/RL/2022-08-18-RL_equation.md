---
layout: single
title: "강화학습 공식 정리"
date: 2022-08-18 04:16:22
lastmod : 2022-08-18 04:16:20
categories: RL
tag: [Q Learning, Sarsa]
toc: true
toc_sticky: true
use_math: true
---

# n-step TD
$G_{t:t+n} \doteq R_{t+1}+\gamma R_{t+2}+ \cdots + \gamma^{n-1}R_{t+n}+\gamma V_{t+n-1}(S_{t+n})$

$V_{t+n}(S_t) \doteq V_{t+n-1}(S_t)+\alpha [G_{t:t+n}-V_{t+n-1}(S_t)]$

# Forward-view TD(λ)
$G_{t:T}^{\lambda}=(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_{t:t+n}+\lambda^{T-t-1}G_{t:T}$

$V_{T}(S_t) = V_{T-1}(S_t)+\alpha_t [G_{t:T}^{\lambda}-V_{T-1}(S_t)]$

**EX1)**

$G_{t:t+1}^{\lambda}=G_{t:t+1}$

$G_{t:t+2}^{\lambda}=(1-\lambda)G_{t:t+1}+\lambda G_{t:t+2}$

$G_{t:t+3}^{\lambda}=(1-\lambda)[G_{t:t+1}+\lambda G_{t:t+2}]+\lambda^2 G_{t:t+3}$

$G_{t:t+n}^{\lambda}=(1-\lambda)[G_{t:t+1}+\lambda G_{t:t+2}+\cdots+\lambda^{n-2}G_{t:t+n-1}]+\lambda^{n-1} G_{t:t+n}$

**EX2**

$G_{t:t+1}=R_{t+1}+\gamma V_t(S_{t+1})$

$G_{t:t+2}=R_{t+1}+\gamma R_{t+2}+\gamma^2 V_{t+1}(S_{t+2})$

$G_{t:t+3}=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\gamma^3 V_{t+2}(S_{t+3})$

$G_{t:t+4}=R_{t+1}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^nV_{t+n-1}(S_{t+n})$

$G_{t:T}=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{T-1}R_T$

# Backward-view TD(λ)

$E_0=0$

$S_t,A_t,R_{t+1},S_{t+1} \sim \pi_{t:t+1}$

$E_t(S_t)=E_t(S_t)+1$

$\delta_{t:t+1}^{\text{TD}}(S_t)=R_{t+1}+\gamma V_t(S_{t+1})-V_t(S_t)$

$V_{t+1}=V_t+\alpha_t \delta_{t:t+1}^{\text{TD}}(S_t)E_t$

$E_{t+1}=E_t\gamma \lambda$

# n-step SARSA
$G_{t:t+n} \doteq R_{t+1}+\gamma R_{t+2}+ \cdots + \gamma^{n-1}R_{t+n}+\gamma Q_{t+n-1}(S_{t+n}, A_{t+n})$

$Q_{t+n}(S_t,A_t) \doteq Q_{t+n-1}(S_t,A_t)+\alpha [G_{t:t+n}-Q_{t+n-1}(S_t,A_t)]$

# SARSA
$Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha_t[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]$

# Q-Learning
$Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha_t[R_{t+1}+\gamma \max_{a}Q(S_{t+1},a)-Q(S_t,A_t)]$


# Actor-Critic

정책 네트워크와 밸류 네트워크를 함께 학습함

## Q Actor-Critic

[Q Actor-Critic pseudo code]
1. 정책 액션-밸류 네트워크의 파라미터 $\theta$와 $w$를 초기화
2. 상태 $s$를 초기화
3. 액션 $a \sim \pi_{\theta}(a|s)$를 샘플링
4. 스텝마다 다음 과정을 반복
    * $a$를 실행하여 보상 $r$과 다음 상태 $s'$를 얻음
    * $\theta$ 업데이트 : $\theta \leftarrow \alpha \nabla_\theta \log \pi_{\theta} (s,a)*Q_w(s,a)$
    * 액션 $a' \sim \pi_{\theta}(a'|s')$
    * $w$ 업데이트 : $w \leftarrow w + \beta(r+\gamma Q_w(s',a')-Q_w(s,a)) \nabla_w Q_w (s,a)$
    * $a \leftarrow a', s \leftarrow s'$

## Advantage Actor-Critic

[Advantage Actor-Critic pseudo code]
1. 3쌍의 네트워크 파라미터 $\theta, w, \phi$를 초기화
2. 상태 $s$를 초기화
3. 액션 $a \sim \pi_{\theta}(a|s)$를 샘플링
4. 스텝마다 다음 과정을 반복
    * $a$를 실행하여 보상 $r$과 다음 상태 $s'$를 얻음
    * $\theta$ 업데이트 : $\theta \leftarrow \theta + \alpha_1 \nabla_{\theta}\log \pi_{\theta}(s,a)*\{Q_w(s,a)-V_{\phi}(s)\}$
    * 액션 $a' \sim \pi_{\theta}(a'|s')$를 샘플링
    * $w$ 업데이트 : $w \leftarrow w+\alpha_2(r+\gamma Q_w(s', a')-Q_w(s,a))\nabla_w Q_w(s,a)$
    * $\phi$ 업데이트 : $\phi \leftarrow \phi + \alpha_3(r+\gamma V_{\phi}(s')-V_{\phi}(s))\nabla_{\phi}V_{\phi}(s)$
    * $a \leftarrow a', s \leftarrow s'$

## TD Actor-Critic

[TD Actor-Critic pseudo code]
1. 정책, 밸류, 네트워크의 파라미터 $\theta$와 $\phi$를 초기화
2. 액션 $a \sim \pi_{\theta}(a|s)$
3. 스텝마다 다음 과정을 반복
    * $a$를 실행하여 보상 $r$과 다음 상태 $s'$를 얻음
    * $\delta$를 계산 : $\delta \leftarrow r+\gamma V_\phi(s')-V_{\phi}(s)$
    * $\theta$ 업데이트 : $\theta \leftarrow \theta+\alpha_1 \nabla_{\theta}\log\pi_{\theta}(s,a)*\delta$
    * $\phi$ 업데이트 : $\phi \leftarrow \phi + \alpha_2 \delta \nabla_{\phi}V_{\phi}(s)$
    * $a \leftarrow a', s \leftarrow s'$


*  : TD Error, 






> 출처
 - 노승은,『바닥부터 배우는 강화 학습』, 영진닷컴(2020)