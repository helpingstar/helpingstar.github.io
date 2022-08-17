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

# SARSA
$Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha_t[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]$

# Q-Learning
$Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha_t[R_{t+1}+\gamma \max_{a}Q(S_{t+1},a)-Q(S_t,A_t)]$