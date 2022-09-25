---
layout: single
title: "pytorch : Categorical, Normal"
date: 2022-09-25 14:42:08
lastmod : 2022-09-25 14:42:06
categories: pytorch
tag: [pytorch, distribution, Categorical, Normal]
toc: true
toc_sticky: true
use_math: true
---

[단단한 심층 강화학습](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791191600674&orderClick=LAG&Kc=)을 보고 아래의 두 코드를 보았다.

코드는 각각 `Categorical`과 `Normal`를 사용하는데, `Categorical`에는 `logits` 인자를 넣는다. 마지막에 `log_prob`를 사용하는데 이 개념들이 이해가 안되서 정리했다.

```python
# Code 2_2
# 정책 네트워크로부터 행동의 로짓 확률을 획득
policy_net_output = torch.tensor([-1.6094, -0.2231])
# pdparams는 로짓으로 probs = [0.2, 0.8]과 동일함
pdparams = policy_net_output
pd = Categorical(logits=pdparams)

# 행동을 추출
action = pd.sample()
print(action)
# => tensor(1) 또는 '오른쪽으로 이동'

# 행동 로그 확률을 계산
print(pd.log_prob(action))
# => tensor(-0.2231)
```

`torch.distributions.categorical.Categorical(probs=None, logits=None, validate_args=None)`

Creates a categorical distribution parameterized by either `probs` or `logits` (but not both).

구현 부분

[torch/distributions/categorical.py](https://github.com/pytorch/pytorch/blob/master/torch/distributions/categorical.py)

```python
...
if (probs is None) == (logits is None):
    raise ValueError("Either `probs` or `logits` must be specified, but not both.")
...
```
위 코드를 통해 `probs`, `logits` 두개 다 있거나 두개 다 없는 상황에 에러를 발생시킨다.

`Categorical(logits=pdparams)` 에서 `logits`에 파라미터를 주게 되면 softmax 연산($\frac{e^{x^{(i)}}}{\sum e^{x^{(i)}}}$)을 통해 확률을 구한다 예시로 `[-1.6094, -0.2231]`의 경우 각각 $e$연산을 하면 각각 `[0.2, 0.8]`이 나오게 되는데 이것은 `prob`에 해당 파라미터를 준 것과 같다. 예를 들어 `logits=[2, 3]`일 경우 $e$ 연산시 각각 `[ 7.3891, 20.0855]` 이 되는데 이는 정규화하여 `prob=[0.2689, 0.7311]`와 같다

`pd.log_prob(action)` : 해당 `action`의 확률의 로그값이다 `torch.log(pd.probs)[action]`와 같다

```python
# Code 2_3
# 하나의 행동을 가정
# 정책 네트워크로부터 행동의 평균과 표준편차를 획득
policy_net_output = torch.tensor([1.0, 0.2])
# pdparams는 (평균, 표준편차) 또는 (loc, scale)
pdparams = policy_net_output
pd = Normal(loc=pdparams[0], scale=pdparams[1])

# 행동을 추출
action = pd.sample()
# => tensor(1.0295), 토크의 크기

# 행동 로그확률을 계산
pd.log_prob(action)
# => tensor(0.6796)
```

`Normal`은 location을 평균으로 하고 scale을 표준편차로 하는 정규분포를 만들고 sample시에는 해당 확률 분포를 토대로 샘플링을 한다.

그런데 연속적일 경우 `log_prob`는 어떻게 계산될까?

[torch/distributions/normal.py](https://github.com/pytorch/pytorch/blob/master/torch/distributions/normal.py)

```python
...
def log_prob(self, value):
    if self._validate_args:
        self._validate_sample(value)
    # compute the variance
    var = (self.scale ** 2)
    log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
    return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
...
```

맨 아랫줄 `return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))`만 보면 되는데

이것은

$$N(x \mid \mu, \sigma^2) \equiv \frac{1}{\sigma \sqrt{2 \pi}}\exp \left [ -\frac{(x-\mu)^2}{2 \sigma^2}\right ]$$

에 로그를 취한 것이다. 위 `return`처럼 나타내보면

$$-\frac{(x-\mu)^2}{2 \sigma^2} - \ln\mu - \ln \sqrt{2\pi}$$

이 된다.


> 출처
 - Laura Graesser, Wah Loon Keng,『단단한 심층 강화학습』, 김성우, 제이펍(2022)
