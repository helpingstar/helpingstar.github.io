---
layout: single
title: "단단한 강화학습 코드 정리, chap8"
date: 2023-06-21 03:17:41
lastmod : 2023-06-21 03:17:41
categories: RL
tag: [Sutton, 단단한 강화학습, RL]
toc: true
toc_sticky: true
use_math: true
---

[ShangtongZhang github](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/tree/master/chapter08)

[단단한 강화학습](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791190665179&orderClick=LAG&Kc=) 책의 코드를 공부하기 위해 쓰여진 글이다.

# Maze

## PriorityQueue

```python
class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder
```
<!-- * **(1)** : 우선순위 큐를 구현한 클래스이다.
* **(2~6)** : 생성자를 정의한다.
  * `self.pq` : 요소를 저장할 리스트
  * `self.entry_finder` : 아이템(key) 우선순위를 위한 entry(value)를 연결하는 dict이다.
  * `self.REMOVED` : 해당 entry가 제거된 것인지 체크하기 위한 문자열
  * self.counter -->


## Maze
```python
# A wrapper class for a maze, containing all the information about the maze.
# Basically it's initialized to DynaMaze by default, however it can be easily adapted
# to other maze
class Maze:
    def __init__(self):
        # maze width
        self.WORLD_WIDTH = 9

        # maze height
        self.WORLD_HEIGHT = 6

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state
        self.START_STATE = [2, 0]

        # goal state
        self.GOAL_STATES = [[0, 8]]

        # all obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.old_obstacles = None
        self.new_obstacles = None

        # time to change obstacles
        self.obstacle_switch_time = None

        # initial state action pair values
        # self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # the size of q value
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))

        # max steps
        self.max_steps = float('inf')

        # track the resolution for this maze
        self.resolution = 1

    # extend a state to a higher resolution maze
    # @state: state in lower resolution maze
    # @factor: extension factor, one state will become factor^2 states after extension
    def extend_state(self, state, factor):
        new_state = [state[0] * factor, state[1] * factor]
        new_states = []
        for i in range(0, factor):
            for j in range(0, factor):
                new_states.append([new_state[0] + i, new_state[1] + j])
        return new_states

    # extend a state into higher resolution
    # one state in original maze will become @factor^2 states in @return new maze
    def extend_maze(self, factor):
        new_maze = Maze()
        new_maze.WORLD_WIDTH = self.WORLD_WIDTH * factor
        new_maze.WORLD_HEIGHT = self.WORLD_HEIGHT * factor
        new_maze.START_STATE = [self.START_STATE[0] * factor, self.START_STATE[1] * factor]
        new_maze.GOAL_STATES = self.extend_state(self.GOAL_STATES[0], factor)
        new_maze.obstacles = []
        for state in self.obstacles:
            new_maze.obstacles.extend(self.extend_state(state, factor))
        new_maze.q_size = (new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions))
        # new_maze.stateActionValues = np.zeros((new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions)))
        new_maze.resolution = factor
        return new_maze

    # take @action in @state
    # @return: [new state, reward]
    def step(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
        else:
            reward = 0.0
        return [x, y], reward
```
* **(1~4)** : 미로를 형성하는 클래스이다.
* **(5)** : 클래스 생성시 실행되는 생성자부분이다.
* **(6~10)** :
  * 너비(가로) : 9
  * 높이(세로) : 6
* **(12~17)** : 위, 아래, 왼쪽, 오른쪽 방향으로 이동하는 행동에 각각 0, 1, 2, 3의 번호를 부여하고 `actions` 변수에 리스트로 저장한다.
* **(19~23)** : 시작상태 : (2, 0), 목표 상태 : (0, 8)
* **(25~28)** : 초기 장애물을 설정한다. `old_obstacles`, `new_obstacles`는 figure 8.4 에서 장애물의 위치를 바꿀때 사용한다.
* **(30~31)** : 장애물을 바꾸게 되는 시간, 해당 시간이 지나면 장애물의 위치가 바뀐다.
* **(36~37)** : Q값을 저장하기 위해 정의된 Q테이블의 크기, (높이, 너비, 행동의 개수) 로 정의된다.
* **(39~40)** : 최대 스텝을 정의한다. 해당 스텝까지만 움직인다.
* **(42~43)** : 미로의 해상도를 나타낸다.
* **(45~48)** : 상태를 더 큰 해상도의 미로에 대응시킨다.
  * `state` : 원래 해상도의 상태
  * `factor` : 확장 인자, 상태는 확장 후에 $\text{(factor)}^{2}$ 개의 상태가 된다.
* **(49)** : 상태를 확장된 크기에 맞게 새로운 위치로 옮긴다.
* **(50)** : 새로운 상태들을 저장할 리스트를 선언한다.
* **(51~53)** : 새로운 위치에서 행, 열 방향으로 `factor` 만큼 확장한다.
  * ex) ![fcode_chap8_1](../../assets/images/rl/fcode_chap8_1.png){: width="50%" height="50%"}
* **(54)** : 새로운 상태들을 리턴한다.
* **(56~58)** : 미로와 미로안의 상태들을 더 큰 해상도로 확장한다.
* **(59~61)** : 새로운 미로를 선언하고 새로운 미로의 너비와 높이를 (기존 크기 × `factor`) 로 저장한다.
* **(62)** : 시작 상태를 새로운 위치에 대응한다. 크기를 넓히지 않고 위치만 넣는다.
* **(63)** : 목표 위치를 `extend_maze` 함수를 이용해 확장하여 저장한다.
* **(64~66)** : 확장된 미로의 장애물을 저장할 리스트를 선언한 후 기존의 장애물을 확장하여 리스트에 추가한다.
* **(67)** : 새로운 미로의 Q 테이블의 크기를 저장한다.
* **(69)** : 새로운 미로의 해상도를 클래스 변수에 저장한다.
* **(70)** : 새롭게 정의된 Maze 클래스를 반환한다.
* **(72~74)** : 상태와 행동을 받아 새로운 상태와 보상을 반환한다.
* **(75)** : 인자로 받은 상태를 x, y에 저장한다.
* **(76~83)** : action에 해당하는 방향으로 움직인다. `max`, `min` 을 통해 미로를 벗어나지 못하게 한다.
* **(84~85)** : 옮긴 위치가 장애물일 경우 원래 위치로 되돌린다.
* **(86~89)** : 옮긴 위치가 목표 위치일 경우 보상을 1로 설정하고 그 외에는 0으로 설정한다.
* **(90)** : 다음 상태와 보상을 반환한다.

## DynaParams
```python
# a wrapper class for parameters of dyna algorithms
class DynaParams:
    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.1

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.time_weight = 0

        # n-step planning
        self.planning_steps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']

        # threshold for priority queue
        self.theta = 0
```

## choose_action
```python
# choose an action based on epsilon-greedy algorithm
def choose_action(state, q_value, maze, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])
```

* **(1~2)** : ε-greedy 정책을 기반으로 하여 행동을 선택한다.
  * `state` : 현재의 상태
  * `q_value` : Q 값이 저장되어 있는 Q 테이블
  * `maze` : 미로정보를 가지고 있는 Maze 클래스 인스턴스
  * `dyna_params` : ε 정보가 저장된 클래스
* **(3~7)** : ε 의 확률로 무작위 행동을 선택하고, 1-ε 의 확률로 가장 Q값이 높은 것들 중에서 무작위로 선택한다. (`np.argmax` 는 가장 큰 값이 여러 개일 경우 인덱스가 낮은 것이 선택된다)

## TrivialModel
```python
# Trivial model for planning in Dyna-Q
class TrivialModel:
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward
```
<!-- * **(1~2)** : Dyna-Q를 planning하기 위한 간단한 모델
* **(3~6)** : // TODO -->

## dyna_q
```python
# play for an episode for Dyna-Q algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
def dyna_q(q_value, model, maze, dyna_params):
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        # track the steps
        steps += 1

        # get action
        action = choose_action(state, q_value, maze, dyna_params)

        # take action
        next_state, reward = maze.step(state, action)

        # Q-Learning update
        q_value[state[0], state[1], action] += \
            dyna_params.alpha * (reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                                 q_value[state[0], state[1], action])

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(0, dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_[0], state_[1], action_] += \
                dyna_params.alpha * (reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) -
                                     q_value[state_[0], state_[1], action_])

        state = next_state

        # check whether it has exceeded the step limit
        if steps > maze.max_steps:
            break

    return steps
```
* **(1~6)** : Dyna-Q 알고리즘을 위해 에피소드를 실행한다.
  * `q_value` : 상태-행동 의 Q값을 저장하는 Q-table
  * `model` : planning을 위한 모델 인스턴스
  * `maze` : 환경의 정보를 저장하는 `Maze` 클래스의 인스턴스
  * `dyna_params` : 알고리즘의 파라미터가 저장된 인스턴스
* **(7)** : 미로의 초기 상태로 현재 상태를 설정한다.
* **(8)** : 현재의 step을 저장할 변수를 0으로 초기화한다.
* **(9)** : 목표 상태(`GOAL_STATES`)에 도달할 때까지 반복한다.
* **(10~11)** : `step` 을 1 증가시킨다. (step이 1부터 시작한다)
* **(13~14)** : 현재 상태, Q값을 가지고 `dyna_params`의 ε값에 따라 maze환경의 행동 중 하나를 선택한다.
* **(16~17)** : 현재 상태에서 행동을 수행하고 다음 상태와 보상을 얻는다.
* **(19~22)** : Q값을 업데이트한다
$$Q(S,A) \leftarrow Q(S,A) + \alpha \left [ R + \gamma \underset{a}{max} Q(S', a) - Q(S,A)\right ]$$
