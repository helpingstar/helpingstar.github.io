---
title: "강화학습 코드/환경 구현시 팁"
date: 2022-10-25 17:19:02
lastmod : 2022-11-10 17:48:38
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

->
내가 본 코드는 [이 코드](https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)였다. 일단 가장 기본적인 해결 방법은 `ReplayBuffert` 수준에서 `Tensor`를 반환하지 않는 것이다. 해당 코드는 학습을 위해 최대한 코드를 줄이느라 그런 것 같다.

그래서 더 좋은 코드가 더 있을까 하고 [stable_baselines3/common/buffers.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)을 봤는데 다음과 같은 방법이 선호되는 듯 하다.

memory를 아끼고 싶다면 맨 처음에 빈 배열로 초기화 되고 memory가 꽉 찰 때까지 `transition`을 추가한다. 아니면 처음부터 최대 버퍼 크기만큼 공간을 확보하고 `transition`을 추가한다.

여기서 추가와 삭제는 pos(커서)와 모듈러(`%`) 연산으로 한다. 모듈러 연산을 통해 메모리를 순환하면서 FIFO로 `transition`을 append하거나 버퍼가 꽉차면 덮어쓴다.

### 2.

`torch.tensor`로 casting할 타이밍을 모르겠다. 어떤 구현에선는 `ReplayBuffer`에서 `numpy`를 `input`으로 받고 `tensor`를 반환한다. 하지만 어떤 구현에서는 `numpy`를 `input`으로 받고 똑같이 `numpy`를 반환한다. 어떤 것이 범용적인지 좀 더 공부와 구현을 해보고 판단해야겠다.

-> 생각보다 간단한 문제였던 것 같다. 너무 `pytorch`기준으로 생각해서 그랬던 것 같다. `tensorflow`를 쓸 수도 있고 `pytorch`를 쓸 수도 있으니 `model`에 넣기 직전에 적절한 `tensor`로 바꿔주는 것이 맞고 전처리나 기타 함수에서는 `numpy`를 받았다면 `numpy`를 반환하는 것이 맞다.

### 3.

강화학습 환경의 `observation_space`에 관한 고민이다. 보드게임 환경을 구현했는데 `(9, 9)` 보드에 블록을 놓으면 되는 게임이어서 (놓지 않음/ 놓음) 은 0과 1로 구별하면 된다. 그리고 블럭 3개도 포함해야 하기 때문에 `observation_space`를 `MultiBinary([15, 15])`로 하였다.

![gym_woodoku_pic_1](../../assets/images/rl/gym_woodoku_pic_1.png){: width="35%" height="35%" class="align-center"}

그런데 내가 공부가 부족해서 그런지 모르겠지만 고민이 많다. `rank=3`로해서 `[15, 15, 1]`로 해야 할지 위와 같이 rank를 2로 할지에 대한 것이다. 알고리즘상 CNN을 통과해야 하기 때문에 `rank=3`인 것이 좋다. 그런데 여기서 고민이 발생한다.

1) `env.observation_space`를 `rank=3`으로 설정하고
   1) 메커니즘을 처음부터 `rank=3`으로 구현하는 지
   2) 계산은 `rank=2`으로 하고 마지막 반환 직전에 `rank=3`으로 변환할지
2) `env.observation_space`를 `rank=2`으로 설정하고
   1) 반환받은 `observation`을 `rank=3`으로 변환 후 CNN에 넣을지

각각의 장단점이 있는 것 같다.

1-1) 방법은 나머지 신경을 안써도 되지만 메커니즘 구현에 있어서 불편하다 `rank`가 하나 늘어버려서 직관적이지 않고 오류를 발생시킬 가능성도 높다.

1-2) 아직은 내 생각에 이 방법이 맞는 것 같다. 구현시에도 직관적이고 강화학습과 연계해서도 구현하기 편해질 것 같다. [gym library](https://www.gymlibrary.dev/environments/atari/)의 atari에서도 컬러여서 그렇긴 하지만 모두 `rank`가 3이니 호환성을 위해서도 3으로 하는게 맞는 것 같다.

2-1) 마음은 편하겠지만, `agent`와 `gym`을 연계시킬 때 `env`의 `space`를 인수로 넣을 수도 있는데 그러면 호환성이 많이 떨어지게 된다.

내 생각은 1-2) 방법이 가장 맞는 것 같다.(2022-11-08)

### 4.

[3번](#3)과 비슷한 문제이다. 방법론적이라거나 테크니컬한 문제는 아니고 일종의 UI/UX 문제인 것 같다. **환경을 만들 때 agent를 위해 어디까지 마중나가야 할까**의 문제이다. 예를 들어 나는 어떤 게임을 구현하기 위해서는 `[9, 9]`의 공간만으로 충분하다고 하자. 이는 `[weight, height]` 라고 할때, 이것을 `CNN`에 적용하고 싶다면 `rank`를 3개로 늘려야 한다. 이는 `[channel, weight, height]`가 될 수도 있고 `[weight, height, channel]`이 될 수도 있다. 내 생각의 발전 순서를 나열해 보겠다.

1. 환경만 신경쓴다. 예를 들어 `[9, 9]` 공간에 0, 1만 필요하다면 `observation_space`를 `MultiBinary([9, 9])`로 정의한다.
2. 대부분 `CNN`은 3개의 `rank`를 필요로 하고 atari같은 경우도 대부분 `[width, height, 3]`을 반환한다. 그러므로 이용자의 편의를 위해 `rank`를 3으로 만드는 것 정도는 해도 되지 않을까 생각했다.
3. `Wrapper`에 대해서 알게 되었고 생각보다 `gym` 라이브러리에 대해서 환경에 대한 수정이 용이하다는 것을 알게 되었다. 그래서 다시 1번과 같은 생각을 하게 되었다.

무한 반복이다. 환경 그 자체로 집중해야 할 지 아니면 그 외의 것도 고려해야할지 고려한다면 어디까지 고려해야 할 지 더 많은 코드를 보고 정해야 겠다.

// TODO
