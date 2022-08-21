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

# n-step SARSA
$G_{t:t+n} \doteq R_{t+1}+\gamma R_{t+2}+ \cdots + \gamma^{n-1}R_{t+n}+\gamma Q_{t+n-1}(S_{t+n}, A_{t+n})$

$Q_{t+n}(S_t,A_t) \doteq Q_{t+n-1}(S_t,A_t)+\alpha [G_{t:t+n}-Q_{t+n-1}(S_t,A_t)]$

# SARSA
$Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha_t[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]$

# Q-Learning
$Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha_t[R_{t+1}+\gamma \max_{a}Q(S_{t+1},a)-Q(S_t,A_t)]$