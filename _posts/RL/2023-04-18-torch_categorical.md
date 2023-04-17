---
layout: single
title: "Pytorch Categorical 이해하기"
date: 2023-04-17 23:32:27
lastmod : 2023-04-17 23:32:29
categories: torch
tag: [torch, distributions, categorical]
toc: true
toc_sticky: true
use_math: true
---

# `Categorical`

`torch.distributions.categorical.Categorical`(*probs*=None, *logits*=None, *validate_args*=None)

Creates a categorical distribution parameterized by either `probs` or `logits` (but not both).

---

`Categorical`은 *logits* 또는 *probs* 인수 중 하나를 받아서 확률 분포를 구성하는 클래스를 만든다는 것이다. 이 두개를 구분하면 어려워진다. 사실 확률을 산출하기 위한 수단일 뿐이다. 하나씩 정복해보자

인수(parameter)와 속성(property)의 단어가 같기 때문에 함수의 파라미터로 쓰이는 단어는 *기울임* 를 적용하고 속성의 경우 `하이라이트`를 적용하여 구분하겠다. 파이토치 내부 구현의 순서가 `prob` -> `logits`는 아니지만 설명의 편의성을 위해 다음 순서를 따르겠다.

## ***probs***

아무런 파라미터도 명시하지 않을 경우 적용되는 파라미터이다. 음수를 입력할 수 없고, 합이 0이 아니어야 한다.

마지막 차원을 따라 1로 합산되도록 정규화된다. `probs`는 이 정규화된 값들을 반환한다.

아래 예시를 위해 다음을 가정해보자

```python
test = torch.tensor([1, 2, 3, 2, 4])
ctest = Cartegorical(probs=test)
```
### `probs`

합이 1이 되도록 정규화한다는 것은 총합으로 나눈다는 뜻이다. 수식으로 하면 다음과 같다.

$$p_i = \frac{x_i}{\sum_i x_i}$$

파이썬 코드로 적어보면 다음과 같다.

```python
test / torch.sum(test)
>>> tensor([0.0833, 0.1667, 0.2500, 0.1667, 0.3333])
```

이것이 호출한 속성값과 같은지 확인해보자

```python
ctest.probs
>>> tensor([0.0833, 0.1667, 0.2500, 0.1667, 0.3333])
```

같은 것을 확인할 수 있다.

이제 이 `probs`를 통해 `logits`를 구해보자 이 다음은 간단하다 각 확률에 $\log_e$를 씌우면 된다

$$l_i = \log_ep_i$$

다음 코드를 통해 확인할 수 있다.

```python
ctest.logits
>>> tensor([-2.4849, -1.7918, -1.3863, -1.7918, -1.0986])
torch.log(ctest.probs)
>>> tensor([-2.4849, -1.7918, -1.3863, -1.7918, -1.0986])
```

그 반대는 다음과 같을 것이다

$$p_i = e^{l_i}$$

코드를 통해 확인해보자
```python
torch.exp(ctest.logits)
>>> tensor([0.0833, 0.1667, 0.2500, 0.1667, 0.3333])
ctest.probs
>>> tensor([0.0833, 0.1667, 0.2500, 0.1667, 0.3333])
```

당연하지만 같은 것을 확인할 수 있다.

## ***logits***

*logits* 인수는 정규화되지 않은 로그 확률로 해석되므로 모든 실수가 될 수 있다. 마찬가지로 정규화되어 결과 확률이 마지막 차원을 따라 1이 되도록 합산된다. `logits`는 이 정규화된 값을 반환한다.

아래 예시를 위해 다음을 가정해보자. 음수도 가능하다는 것을 보이기 위해 음수도 추가하였다.

```python
test = torch.tensor([-1, -2, 3, 2, 4])
ctest = Categorical(logits=test)
```

### `probs`

*logits* 를 인수로 받았을 때 `prob`를 계산해보자. 수식은 다음과 같다. softmax 연산과 같다.

$$p_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$$

파이썬 코드로 적어보면 다음과 같다.

```python
torch.exp(test) / torch.sum(torch.exp(test))
>>> tensor([0.0045, 0.0016, 0.2432, 0.0895, 0.6612])
```
이것이 호출한 속성값과 같은지 확인해보자

```python
ctest.probs
>>> tensor([0.0045, 0.0016, 0.2432, 0.0895, 0.6612])
```

`probs`를 통해 `logits`를 얻는 것은 위에서도 설명했지만 같은 서술로 똑같이 적용해보겠다.

이제 이 `probs`를 통해 `logits`를 구해보자 이 다음은 간단하다 각 확률에 $\log_e$를 씌우면 된다

$$l_i = \log_ep_i$$

다음 코드를 통해 확인할 수 있다.

```python
ctest.logits
>>> tensor([-5.4137, -6.4137, -1.4137, -2.4137, -0.4137])
torch.log(ctest.probs)
>>> tensor([-5.4137, -6.4137, -1.4137, -2.4137, -0.4137])
```

그 반대는 다음과 같을 것이다

$$p_i = e^{l_i}$$

코드를 통해 확인해보자
```python
torch.exp(ctest.logits)
>>> tensor([0.0045, 0.0016, 0.2432, 0.0895, 0.6612])
ctest.probs
>>> tensor([0.0045, 0.0016, 0.2432, 0.0895, 0.6612])
```

당연하지만 같은 것을 확인할 수 있다.


## `entropy()`

이제 `Categorical` 클래스가 확률을 어떻게 가지는지 확인해보았을 것이다. 이 확률을 통해 해당 확률 분포의 엔트로피를 구할 수 있다. 

엔트로피 공식은 다음과 같다. (밑은 $e$로 한다)

$$H = -\sum_i p_i \log_e(p_i)$$

이제 확률 얻는 법을 알았으니 예시는 하나만 들겠다. 아래 예시를 들어보겠다.

```python
test = torch.tensor([1, 2, 3, 2, 4])
ctest = Cartegorical(probs=test)

ctest.probs
>>> tensor([0.0833, 0.1667, 0.2500, 0.1667, 0.3333])

ctest.logits
>>> tensor([-2.4849, -1.7918, -1.3863, -1.7918, -1.0986])
```

눈치챈 분도 있겠지만 `probs`와 `logits`를 요소별로 곱한 후 모두 더한 후에 마이너스를 씌우면 된다. 아래 다양한 예제로 이를 이해해보자

```python
ctest.entropy()
>>> tensor(1.5171)

-torch.sum(ctest.probs * ctest.logits)
>>> tensor(1.5171)

torch.sum(torch.log(ctest.probs) * ctest.probs) * (-1)
>>> tensor(1.5171)

-torch.sum(ctest.probs.dot(ctest.logits))
>>> tensor(1.5171)
```