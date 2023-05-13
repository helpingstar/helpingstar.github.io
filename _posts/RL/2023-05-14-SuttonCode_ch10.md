---
layout: single
title: "단단한 강화학습 코드 정리, chap10"
date: 2023-05-14 02:09:39
lastmod : 2023-05-14 02:09:36
categories: RL
tag: [Sutton, 단단한 강화학습, RL]
toc: true
toc_sticky: true
use_math: true
---

[ShangtongZhang github](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/tree/master/chapter10)

[단단한 강화학습](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791190665179&orderClick=LAG&Kc=) 책의 코드를 공부하기 위해 쓰여진 글이다.

# Tile Coding
```python
# Following are some utilities for tile coding from Rich.
# To make each file self-contained, I copied them from
# http://incompleteideas.net/tiles/tiles3.py-remove
# with some naming convention changes
```
## IHT
```python
class IHT:
    "Structure to handle collisions"
    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count
```
* **(1)** : 타일 코딩을 위한 해시값을 구하는 함수
* **(3~6)** : 클래스의 생성자
  * `self.size = size_val` : 인덱스의 최댓값(=인덱스의 개수)
  * `self.overfull_count` : // TODO
  * `self.dictionary` : // TODO

## hash_coords
```python
def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT): return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int): return hash(tuple(coordinates)) % m
    if m is None: return coordinates
```
* **(1)** : 해시값을 반환하는 함수이다
  * `coordinates` : 해시값을 산출하기 위한 정수 리스트
  * `m` : 해시의 종류, `IHT` 클래스의 인스턴스 혹은 정수값이다.
  * `read_only` : True일 경우 읽기만 하며 해시테이블에 key 값이 없으면 None을 반환한다. False일 경우 key 값에 해당하는 Value 값을 만들고 해당 값을 해시테이블에 저장하고 해당 값을 반환한다..
* **(2)** : `m` 이 [`IHT`](#iht) 클래스의 인스턴스일 경우 `get_index` 메소드를 이용하여 해시값을 반환한다.
* **(3)** : `m` 이 정수일 경우 `coordinates`의 튜플값의 해시값(파이썬 해시함수 사용)의 나머지를 반환한다.
* **(4)** : `m` 이 `None`일 경우 `coordinates`를 그대로 반환한다.

## tiles
```python
def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    tiles = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tiles.append(hash_coords(coords, iht_or_size, read_only))
    return tiles
```
* **(1~2)** : 수들을 받고 그에 해당하는 tile 인덱스들을 반환하는 함수
  * `iht_or_size` : [`IHT`](#iht) 클래스 또는 정수값을 인수로 받아 [`hash_coords`](#hash_coords)에서 해당 인스턴스에 해당하는 해시값을 구한다.
  * `num_tilings` : 반환하는 타일 인덱스의 개수
  * `floats` : 상태를 표현하는 실수, 여기서는 (위치, 속도)로 표현된다.
  * `ints` : 행동을 표현하는 정수, 여기서는 -1, 0, 1 중 하나가 된다.
  * `read_only` : True일 경우 읽기만 하며 해시테이블에 key 값이 없으면 None을 반환한다. False일 경우 key 값에 해당하는 Value 값을 만들고 해당 값을 해시테이블에 저장하고 해당 값을 반환한다..
* **(3~4)** : `ints` 가 `None`일 경우 `ints`를 빈 리스트로 초기화한다. 상태 가치만 얻고 싶다면 `ints`를 `None`으로 하여 근사할 수 있다. 예시에서는 행동도 같이 제공하므로 해당하지 않는다.
* **(5)** : 상태를 나타내는 `floats`의 실수값들에 `num_tilings`를 곱한 후 내림하여 정수로 만든다. quantize의 q가 아닐까 싶다.
* **(6)** : 반환할 tile의 인덱스를 저장하는 빈 리스트를 생성한다. 
* **(7)** : `num_tilings` 횟수만큼 반복하며 반복마다 한 해시값을 `tiles`에 추가한다. `tiling`은 반복 변수이다.
* **(8)** : 0부터 시작하는 반복변수 `tiling`에서 2를 곱한 값을 저장한다.
* **(9)** : `coords` 리스트에 반복변수 `tiling`을 넣고 초기화한다.
* **(10)** : `b`에 반복변수 `tiling`을 저장한다.
* **(11~13)** : 각 `qfloats`의 값(상태에 타일의 개수를 곱한 후 내림한 값)에 b를 더하고 타일의 개수로 나눈후 내림한 값을 `coords` 리스트에 추가한다. b에 `tilingX2`(`tiling` 반복변수에 2를 곱한 값)를 더한다.
* $$ K_j =\lfloor \frac{Q_j + (i + 2 \times (j-1))}{\vert T \vert} \rfloor$$
  * $K$ : coords에 추가되는 숫자들
  * $\vert T \vert$ : number of tiles
  * $Q_j$ : $\lfloor S_j \times \vert T \vert \rfloor$
  * $i$ : index of tile
  * $j$ : index of state
* **(14)** : `coords`에 `ints`를 추가한다. 이러면 `coords`는 [index of tile] + $K$ + [ints] 가 된다.
* **(15)** : `coords`, `iht_or_size`, `read_only` 값을 기반으로 얻은 해시값을 `tiles` 리스트에 추가한다.
* **(16)** : `num_tilings` 개수 만큼 타일 인덱스를 가진 `tiles`를 반환한다. 



# ValueFunction
```python
# wrapper class for state action value function
class ValueFunction:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @max_size: the maximum # of indices
    def __init__(self, step_size, num_of_tilings=8, max_size=2048):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings

        # divide step size equally to each tiling
        self.step_size = step_size / num_of_tilings

        self.hash_table = IHT(max_size)

        # weight for each tile
        self.weights = np.zeros(max_size)

        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_of_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    # get indices of active tiles for given state and action
    def get_active_tiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        active_tiles = tiles(self.hash_table, self.num_of_tilings,
                            [self.position_scale * position, self.velocity_scale * velocity],
                            [action])
        return active_tiles

    # estimate the value of given state and action
    def value(self, position, velocity, action):
        if position == POSITION_MAX:
            return 0.0
        active_tiles = self.get_active_tiles(position, velocity, action)
        return np.sum(self.weights[active_tiles])

    # learn with given state, action and target
    def learn(self, position, velocity, action, target):
        active_tiles = self.get_active_tiles(position, velocity, action)
        estimation = np.sum(self.weights[active_tiles])
        delta = self.step_size * (target - estimation)
        for active_tile in active_tiles:
            self.weights[active_tile] += delta

    # get # of steps to reach the goal under current state value function
    def cost_to_go(self, position, velocity):
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        return -np.max(costs)
```
* **(1~7)** : 상태 행동 가치 함수를 표현하기 위한 클래스이다. 타일링은 (state, action)를 인덱스 시리즈에 대응시킨다. 대응이 몇개의 속성을 만족한다면 인덱스가 의미를 가지는지는 중요하지 않다.
* **(8~9)** : 클래스의 생성자이다.
  * `step_size` : 시간 간격을 의미하며 $\alpha$로 표기된다.
  * `num_of_tilings` : (state, action)마다 대응되는 인덱스의 개수
  * `max_size` : 인덱스의 최댓값(=개수)
* **(10~11)** : 인수로 들어온 `num_of_tilings`, `max_size`를 클래스 변수에 저장한다.
* **(13~14)** : 인수로 들어온 `step_size`를 타일의 개수로 나누어 클래스 변수에 저장한다.
* **(16)** : 타일링을 위한 해시 값을 구하기 위한 IHT(Index Hash Table) 클래스를 선언하고 클래스변수에 저장한다.


# figure_10_2
```python
# Figure 10.2, semi-gradient Sarsa with different alphas
def figure_10_2():
    runs = 10
    episodes = 500
    num_of_tilings = 8
    alphas = [0.1, 0.2, 0.5]

    steps = np.zeros((len(alphas), episodes))
    for run in range(runs):
        value_functions = [ValueFunction(alpha, num_of_tilings) for alpha in alphas]
        for index in range(len(value_functions)):
            for episode in tqdm(range(episodes)):
                step = semi_gradient_n_step_sarsa(value_functions[index])
                steps[index, episode] += step

    steps /= runs
```

* **(1~2)** : 