---
layout: single
title: "딥러닝 활성화 함수"
date: 2022-07-03 01:28:01
lastmod : 2022-07-04 15:16:25
categories: ML
tag: [RMSE, MAE]
toc: true
toc_sticky: true
use_math: true
---

# **시그모이드 함수, sigmoid**

$\sigma (t)=\frac{1}{1+exp(-t)}$

![sigmoid_function](../../../assets/images/ai/sigmoid_function.png){: width="75%" height="75%" class="align-center"}

# 초기화 전략

`Xavier initialization`, `Glorot initialization`

세이비어 글로럿(Xavier Glorot)과 요슈아 벤지오(Yoshua Bengio)는 논문<sup>[1](#footnote_1)</sup>에서 당시 그레디언트를 불안정하게 만드는 원인에 대해 의심되는 원인을 몇가지 발견하였다.

`sigmoid` 활성화 함수와 정규분포로 가중치를 초기화하는 방식을 사용했을 때 각 층에서 출력의 분산이 입력의 분산보다 더 크다는 것을 발견했다. 신경망이 위쪽으로 갈수록 층을 지날 때마다 분산이 계속 커져 가장 높은 층에서는 활성화 함수가 0이나 1로 수렴한다. 이는 로지스틱 함수의 평균이 0이 아니고 0.5라는 사실때문에 더욱 그렇다. 로지스틱 함수는 항상 양수를 출력하므로 출력의 가중치 합이 입력보다 커질 가능성이 높다. (편향 이동)

로지스틱 활성화 함수를 보면 입력이 커지면 0이나 1로 수렴하여 기울기가 0과 가까워져 역전파가 될 때 사실상 신경망으로 전파할 그레이디언트가 거의 없어지게 된다.

글로럿과 벤지오는 논문에서 불안정한 그레이디언트 문제를 크게 완화하는 방법을 제안한다. 

예측을 할 때는 정방향으로, 그레이디언트를 역전파할 때는 역방향으로 양방향 신호가 적절하게 흘러야 한다. 신호가 죽거나 폭주 또는 소멸하지 않아야 한다

저자들은 적절한 신호가 흐르기 위해서는 각 층의 출력에 대한 분산이 입력의 분산과 같아야 한다고 주장한다. 그리고 역방향에서 층을 통과하기 전과 후의 그레이디언트 분산이 동일해야 한다. 

사실 층의 입력(fan-in)과 출력(fan-out) 연결 개수가 같지 않다면 이 두 가지를 보장할 수 없다.

그에 대해 대안을 제시했는데 아래의 식의 방식대로 무작위로 초기화 하는 것이다. 

![sigmoid_function](../../../assets/images/ai/glorot_xavier_init.jpg){: width="75%" height="75%" class="align-center"}

위 식에서 $fan_{avg}$를 $fan_{in}$로 바꾸면 르쿤(LeCun) 초기화라고 부른다(1990) $fan_{in}=fan_{avg}$이면 르쿤 초기화는 글로럿 초기화와 동일하다. 

글로럿 초기화를 사용하면 훈련 속도를 상당히 높일 수 있다.

일부 논문<sup>[2](#footnote_2)</sup>들이 다른 활성화 함수에 대해 비슷한 전략을 제안했다.

| Initialization | Activation functions          | $\sigma^{2}$(Normal) |
|----------------|-------------------------------|----------------------|
| Glorot         | None, tanh, logistic, softmax | $1/fan_{avg}$        |
| He(kaiming)             | ReLU and variants             | $2/fan_{in}$         |
| LeCun          | SELU                          | $1/fan_{in}$         |

위 표에서 보이듯이 $fan_{avg}$ 또는 $fan_{in}$을 쓰는 것만 다르다. 균등 분포의 경우 단순히 $r=\sqrt{3\sigma^{2}}$로 계산한다. `ReLU` 활성화 함수 및 그의 변종들에 대한 초기화 전략을 논문 저자의 이름을 따서 `He(kaiming) initialization` 이라고 부른다. 뒤에 나오지만 `SELU`는 르쿤 초기화를 사용한다.

## 파이토치의 경우
[`Linear`](https://github.com/pytorch/pytorch/blob/caa6ef15a294c96fad3bf67a10a8b4fa605080bb/torch/nn/modules/linear.py#L103-L111)와 [`conv2d`](https://github.com/pytorch/pytorch/blob/caa6ef15a294c96fad3bf67a10a8b4fa605080bb/torch/nn/modules/conv.py#L146-L155)의 경우 해당 소스코드를 보면 확인할 수 있다.

```python
# pytorch/torch/nn/modules/linear.py

class Linear(Module):
    ...
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
```
```python
# pytorch/torch/nn/modules/conv.py

class _ConvNd(Module):
    ...
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
```

보다시피 `self.weight`에 대해 공통적으로 `kaiming_uniform_`을 적용하는 것을 볼 수 있다. 이 함수를 따라가보자

[`kaiming_uniform_`](https://github.com/pytorch/pytorch/blob/caa6ef15a294c96fad3bf67a10a8b4fa605080bb/torch/nn/init.py#L366)은 다음과 같이 정의되어 있다.
```python
# pytorch/torch/nn/init.py

def kaiming_uniform_(tensor: Tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    ...
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            kaiming_uniform_,
            (tensor,),
            tensor=tensor,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity)

    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0) * std  
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
```

 - `_calculate_correct_fan`은 `mode` 변수의 값에 따라 `fan_in` 또는 `fan_out`을 반환한다. 그런데 여기서는 인자를 주지 않았으므로 기본값인 `fan_in`을 반환하게된다
 - [gain](#calculate_gain) : 여기서는 `calcuate_gain` 함수를 호출하는데 (더 자세한 내용은 아래 서술한다 일단 함수에 집중해보자) `nonlinearity=leaky_relu`, `a=math.sqrt(5)`이므로 `negative_slope=math.sqrt(5)`에 해당하여 $\sqrt{2/(1+negative\_slope^{2})}=1/\sqrt{3}$를 반환하여 `gain`$=1/\sqrt{3}$이 된다.
 - `std` = $gain / \sqrt{fan}$ : `Linear`, `_ConvNd`의 경우 `mode` 파라미터에 인자를 주지 않았으므로 $fan=fan_{in}$이다.
 - `bound` = $\sqrt{3(std)}$
 - `return` : $\plusmn bound$ 에 대한 균등 분포가 반환된다.

위에서 서술했던 `He initialization`(=`kaiming init`)인 $r=\sqrt{3\sigma^{2}}$에 대해 $\plusmn r$을 범위로 하는 균등분포를 쓰는 것이 확인되었다.

## `calculate_gain`

아래 서술되는 `non-linearity`는 `non-linear function`를 의미하는데 편의를 위해 사실상 딥러닝에서의 활성화 함수라고 생각하면 될 것이다.

`gain`은 초기화 함수에 대한 scaling factor라고 한다.<sup>[3](#footnote_3)</sup> `non-linearity`에 적용하는 표준편차를 scale하기 위해 사용된다. `non-linearity`가 활성화의 표준편차에 영향을 주기 때문에 기울기 소실같은 문제가 발생할 수 있다. `non-linearity`에 대한 `gain`은 활성화에 대한 "좋은" 통계를 제공해야 한다.

적절한 `gain` 값으로는 다음과 같다.
| non-linearity    | gain                             |
|-----------------|----------------------------------|
| `Linear/Identity` | $1$                            |
| `Conv{1,2,3}D`    | $1$                            |
| `Sigmoid`         | $1$                             |
| `Tanh`            | $5/3$                            |
| `ReLU`            | $\sqrt{2}$                       |
| `Leaky Relu`      | $\sqrt{2/1+negative\_slope^{2}}$ |
| `SELU`            | $3/4$                            |

[`calculate_gain`](https://github.com/pytorch/pytorch/blob/caa6ef15a294c96fad3bf67a10a8b4fa605080bb/torch/nn/init.py#L67) 함수는 다음과 같이 정의되어있다.
```python
# pytorch/torch/nn/init.py 

def calculate_gain(nonlinearity, param=None):
    ...
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
        return 3.0 / 4  
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
```
보다시피 위에서 설명했던 `gain`값을 얻는 과정을 나타내고 있으며 `SELU`의 경우 `gain`값이 경험적으로 얻어졌음을 나타내고 있다.


<a name="footnote_1">1</a>: [Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks."](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

<a name="footnote_2">2</a>: [Kaiming He, "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"](https://arxiv.org/pdf/1502.01852.pdf)

<a name="footnote_3">3</a>: https://discuss.pytorch.org/t/what-is-gain-value-in-nn-init-calculate-gain-nonlinearity-param-none-and-how-to-use-these-gain-number-and-why/28131

> 출처
 - Aurelien, Geron, 『핸즈온 머신러닝』, 박해선, 한빛미디어(2020)