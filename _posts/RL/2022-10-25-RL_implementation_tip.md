---
title: "강화학습 구현시 팁"
date: 2022-10-25 17:19:02
lastmod : 2022-10-25 17:19:05
categories: RL
tag: [RL]
toc: true
toc_sticky: true
use_math: true
---

강화학습 알고리즘/논문 구현시 겪었던 고충들을 늘어놓고 해결할 때마다 업데이트 하기 위한 글이다. 지속적으로 업데이트 할 예정이다.

### 1.

`state`는 대부분 `np.array`인데 이것을 `ReplayBuffer`에 저장하기 위해서 `list`에 저장하면 `torch.tensor`로 옮길 때 옮겨지긴 하지만 `Warning`이 발생한다.

```
UserWarning:
Creating a tensor from a list of numpy. ndarrays is extremely slow.
Please consider converting the list to a single numpy.
ndarray with numpy.array() before converting to a tensor.
```

그러면 2차원으로 단순히 하면되는 것 아닌가? 하겠지만 한 `transition`에는 `(s, a, r, s', done)`이 포함되기 때문에 단순히 `state`만 볼 수 없다.

// TODO

### 2.

`torch.tensor`로 casting할 타이밍을 모르겠다. 어떤 구현에선는 `ReplayBuffer`에서 `numpy`를 `input`으로 받고 `tensor`를 반환한다. 하지만 어떤 구현에서는 `numpy`를 `input`으로 받고 똑같이 `numpy`를 반환한다. 어떤 것이 범용적인지 좀 더 공부와 구현을 해보고 판단해야겠다.

-> 생각보다 간단한 문제였던 것 같다. 너무 `pytorch`기준으로 생각해서 그랬던 것 같다. `tensorflow`를 쓸 수도 있고 `pytorch`를 쓸 수도 있으니 `model`에 넣기 직전에 적절한 `tensor`로 바꿔주는 것이 맞고 전처리나 기타 함수에서는 `numpy`를 받았다면 `numpy`를 반환하는 것이 맞다.
