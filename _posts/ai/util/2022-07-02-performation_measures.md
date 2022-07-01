---
layout: single
title: "머신러닝/딥러닝 성능 지표"
date: 2022-07-02 00:21:19
lastmod : 2022-07-02 00:21:23
categories: EffectiveCpp
tag: [cpp, c++, new, delete]
toc: true
toc_sticky: true
---

# **오차 행렬**

![confusion_matrix](../../../assets/images/ai/confusion_matrix.jpg)

 - **정밀도**(precision) = $\frac{TP}{TP+FP}$
   - Positive 로 분류된 것 중에 실제로 Positive인 비율
 - 진짜 양성 비율(TPR) = **재현율**(recall) = $\frac{TP}{TP+FN}$
   - 실제 Positive중에 Positive로 분류된 비율
 - 거짓 양성 비율(FPR) = $\frac{FP}{FP+TN}$
   - 실제 Negative중에 Positive로 분류된 비율
 - 진짜 음성 비율(TNR) = 특이도(specificity) = $\frac{TN}{FP+TN}$
   - 실제 Negative중에 Negative으로 분류된 비율

$FPR=\frac{FP}{FP+TN}=1-TNR=1-\frac{TN}{FP+TN}$

# **`f1-score`**
$F_{1}=\frac{2}{\frac{1}{precision}+\frac{1}{recall}}=2\times\frac{preicision\times recall}{preicison+recall}=\frac{TP}{TP+\frac{FN+FP}{2}}$









Aurelien, Geron, 『핸즈온 머신러닝』, 박해선, 한빛미디어(2020)