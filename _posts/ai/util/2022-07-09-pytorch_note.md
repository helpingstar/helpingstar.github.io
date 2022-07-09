---
layout: single
title: "파이토치 메모장"
date: 2022-07-09 18:03:54
lastmod : 2022-07-09 18:03:50
categories: pytorch
tag: [CG, DCG, NDCG, RecSys, 추천 시스템]
toc: true
toc_sticky: true
use_math: true
---

# `torch.backends.cudnn.benchmark`
**What does torch.backends.cudnn.benchmark do?**

This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

cudnn의 auto-tuner이 내 하드웨어에 가장 좋은 알고리즘을 찾아 적용하여 속도를 더 빠르게 해주는 것 같다.
