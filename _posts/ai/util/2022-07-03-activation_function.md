---
layout: single
title: "머신러닝 회귀 성능지표"
date: 2022-07-03 01:28:01
lastmod : 2022-07-03 01:28:04
categories: ML
tag: [RMSE, MAE]
toc: true
toc_sticky: true
use_math: true
---

# **시그모이드 함수, sigmoid function**

$\sigma (t)=\frac{1}{1+exp(-t)}$

![sigmoid_function](../../../assets/images/ai/sigmoid_function.png){: width="75%" height="75%"}

세이비어 글로럿(Xavier Glorot)과 요슈아 벤지오(Yoshua Bengio)는 논문<sup>[1](#footnote_1)</sup>에서 당시 그레디언트를 불안정하게 만드는 원인에 대해 의심되는 원인을 몇가지 발견하였다.

sigmoid 활성화 함수와 정규분포로 가중치를 초기화하는 방식을 사용했을 때 각 층에서 출력의 분산이 입력의 분산보다 더 크다는 것을 발견했다. 신경망이 위쪽으로 갈수록 층을 지날 때마다 분산이 계속 커져 가장 높은 층에서는 활성화 함수가 0이나 1로 수렴한다. 이는 로지스틱 함수의 평균이 0이 아니고 0.5라는 사실때문에 더욱 그렇다. 로지스틱 함수는 항상 양수를 출력하므로 출력의 가중치 합이 입력보다 커질 가능성이 높다. (편향 이동)

로지스틱 활성화 함수를 보면 입력이 커지면 0이나 1로 수렴하여 기울기가 0과 가까워져 역전파가 될 때 사실상 신경망으로 전파할 그레이디언트가 거의 없어지게 된다.

글로럿과 벤지오는 논문에서 불안정한 그레이디언트 문제를 크게 완화하는 방법을 제안한다. 

예측을 할 때는 정방향으로, 그레이디언트저자들은 적절한 신호가 흐르기 위해서는 각 층의 출력에 대한 분산이 입력의 분산과 같아야 한다고 주장한다. 그리고 역방향에서 층을 통과하기 전과 후의 그레이디언트 분산이 동일해야 한다. 

<a name="footnote_1">1</a>: [Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010.](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

> 출처
 - Aurelien, Geron, 『핸즈온 머신러닝』, 박해선, 한빛미디어(2020)