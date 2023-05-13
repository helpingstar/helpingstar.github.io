---
layout: single
title: "단단한 강화학습 코드 정리, chap5"
date: 2023-05-10 18:51:36
lastmod : 2023-05-13 11:54:20
categories: RL
tag: [Sutton, 단단한 강화학습, RL]
toc: true
toc_sticky: true
use_math: true
---

[ShangtongZhang github](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/tree/master/chapter05)

[단단한 강화학습](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791190665179&orderClick=LAG&Kc=) 책의 코드를 공부하기 위해 쓰여진 글이다.

# blackjack

```python
# actions: hit or stand
ACTION_HIT = 0
ACTION_STAND = 1  #  "strike" in the book
ACTIONS = [ACTION_HIT, ACTION_STAND]

# policy for player
POLICY_PLAYER = np.zeros(22, dtype=np.int)
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT
POLICY_PLAYER[20] = ACTION_STAND
POLICY_PLAYER[21] = ACTION_STAND

# function form of target policy of player
def target_policy_player(usable_ace_player, player_sum, dealer_card):
    return POLICY_PLAYER[player_sum]

# function form of behavior policy of player
def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STAND
    return ACTION_HIT

# policy for dealer
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STAND

# get a new card
def get_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# get the value of a card (11 for ace).
def card_value(card_id):
    return 11 if card_id == 1 else card_id
```

* **(1~4)** : 행동의 종류
  * `ACTION_HIT` : 0, 카드를 더 받는다.
  * `ACTION_STAND` : 1, 카드를 그만 받는다., 책에서는 건너뛰기(stick)라고 써있다.
  * `ACTIONS` : 행동을 담은 리스트
* **(6~11)** : 플레이어의 정책
  * 19 이하 : hit, 0으로 초기화된 배열이기 때문에 코드상 의미없다. 11이하는 무조건 hit이기 때문에 따로 적지 않은 듯 하다.
  * 20 이상 : stand
* **(13~15)** : **(6~11)** 에서 작성한 목표 정책(target policy), 플레이어의 카드의 합(`player_sum`)에 따라 정책대로 행동한다. 다른 변수들이 있지만 사용되지는 않는데, 사용자가 임의로 수정할 수 있도록 배려한 듯 하다
* **(17~21)** : 행동 정책, 각 50%의 확률로 hit, stand를 결정한다.
* **(23~28)** : 딜러의 정책(블랙잭 룰)
  * 16 이하 : hit
  * 17 이상 : stand
* **(30~34)** : 카드를 뽑는 함수이다. 2~10, AJQK의 13개의 카드중 하나를 무작위로 뽑고 J, Q, K는 10으로 취급되므로 최댓값을 10으로 클리핑하고 숫자를 리턴한다. 카드 중복은 고려하지 않는 것 같다.
* **(36~38)** : `card_id`를 받고 카드의 가치를 반환하는 함수인데 11이 될 수 있는 A를 고려하기 위함이다. `card_id`가 1인 경우 11을 반환하고 이외는 `card_id`를 반환한다.

## play
```python
# play a game
# @policy_player: specify policy for player
# @initial_state: [whether player has a usable Ace, sum of player's cards, one card of dealer]
# @initial_action: the initial action
def play(policy_player, initial_state=None, initial_action=None):
    # player status

    # sum of player
    player_sum = 0

    # trajectory of player
    player_trajectory = []

    # whether player uses Ace as 11
    usable_ace_player = False

    # dealer status
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False

    if initial_state is None:
        # generate a random initial state

        while player_sum < 12:
            # if sum of player is less than 12, always hit
            card = get_card()
            player_sum += card_value(card)

            # If the player's sum is larger than 21, he may hold one or two aces.
            if player_sum > 21:
                assert player_sum == 22
                # last card must be ace
                player_sum -= 10
            else:
                usable_ace_player |= (1 == card)

        # initialize cards of dealer, suppose dealer will show the first card he gets
        dealer_card1 = get_card()
        dealer_card2 = get_card()

    else:
        # use specified initial state
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()

    # initial state of the game
    state = [usable_ace_player, player_sum, dealer_card1]

    # initialize dealer's sum
    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2)
    # if the dealer's sum is larger than 21, he must hold two aces.
    if dealer_sum > 21:
        assert dealer_sum == 22
        # use one Ace as 1 rather than 11
        dealer_sum -= 10
    assert dealer_sum <= 21
    assert player_sum <= 21

    # game starts!

    # player's turn
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            # get action based on current sum
            action = policy_player(usable_ace_player, player_sum, dealer_card1)

        # track player's trajectory for importance sampling
        player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])

        if action == ACTION_STAND:
            break
        # if hit, get new card
        card = get_card()
        # Keep track of the ace count. the usable_ace_player flag is insufficient alone as it cannot
        # distinguish between having one ace or two.
        ace_count = int(usable_ace_player)
        if card == 1:
            ace_count += 1
        player_sum += card_value(card)
        # If the player has a usable ace, use it as 1 to avoid busting and continue.
        while player_sum > 21 and ace_count:
            player_sum -= 10
            ace_count -= 1
        # player busts
        if player_sum > 21:
            return state, -1, player_trajectory
        assert player_sum <= 21
        usable_ace_player = (ace_count == 1)

    # dealer's turn
    while True:
        # get action based on current sum
        action = POLICY_DEALER[dealer_sum]
        if action == ACTION_STAND:
            break
        # if hit, get a new card
        new_card = get_card()
        ace_count = int(usable_ace_dealer)
        if new_card == 1:
            ace_count += 1
        dealer_sum += card_value(new_card)
        # If the dealer has a usable ace, use it as 1 to avoid busting and continue.
        while dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1
        # dealer busts
        if dealer_sum > 21:
            return state, 1, player_trajectory
        usable_ace_dealer = (ace_count == 1)

    # compare the sum between player and dealer
    assert player_sum <= 21 and dealer_sum <= 21
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory
```

* **(1~5)** : 게임을 실행하는 함수
  * `policy_player` : 플레이어의 정책함수
  * `initial_state` : 게임의 초기 상태
  * `initial_action` : 게임 실행시 초기 행동
* **(6)** : 플레이어의 초기상태를 설정한다.
* **(8~9)** : 플레이어의 숫자의 합
* **(11~12)** : 플레이어의 play trajectory를 저장하는 리스트, 각각은 (상태, 행동) 형태로 저장되며 상태는 (에이스 사용 가능 여부, 숫자의 합, 딜러카드 1개)로 되어있다.
* **(14~15)** : 플레이어가 에이스를 가지고 있는지(=에이스를 11로 사용하는지 여부), `False`로 초기화
* **(17)** : 딜러의 초기상태를 설정한다.
* **(18~19)** : 딜러의 상태, 카드 두 장을 나타내는 변수 두 개를 0으로 초기화한다.
* **(20)** : 딜러가 에이스를 가지고 있는지, `False`로 초기화
* **(22~23)** : `initial_state`가 `None`이면(초기상태가 지정되어 있지 않으면) 초기상태를 만든다.
* **(25~26)** : 카드의 합이 11이하인 상태에서만 hit를 한다(=카드를 뽑는다.)
* **(27~28)** : 카드를 뽑아서 `card` 변수에 저장하고 에이스 여부를 판단한 뒤에 `player_sum`에 누적한다.
* **(30~31)** : 만약 11이하에서 어떤 카드를 더해서 `player_sum`이 21보다 커졌다면 그수는 11일 것이고 에이스를 뽑았다는 뜻이다.
* **(32)** : 프로그램을 제대로 짰다면 `player_sum`은 22가 되어야 하니 이외의 경우는 `assert`를 한다.
* **(33~34)** : `player_sum`이 21을 넘었으므로 뽑은 에이스를 11에서 1로 바꾼다.
* **(35~36)** : 21을 넘지 않고 뽑은 카드가 1일 경우 `usable_ace_player`를 `True`로 바꾼다. (`|` : or연산)
* **(38~40)** : 딜러가 카드를 두 장 뽑고 첫 번째 카드를 공개한다.
* **(42~45)** : `initial_state`가 명시되어 있을 경우 `initial_state`로 상태를 초기화하고 딜러의 두번째 카드를 뽑는다.
* **(47~48)** : 위에서 구한 초기 상태를 `state` 변수에 저장한다.
* **(50~51)** : 딜러의 카드 두 장의 합을 계산한다.
* **(52)** : 딜러 카드 중 1이 있으면 `usable_ace_dealer`를 `True`로 저장한다.
* **(53~57)** : 딜러의 두 카드의 합이 21을 넘으면 에이스가 두 개 있다는 뜻이므로 한 에이스를 사용하여 1로 만든다(=합에서 10을 뺀다.).
* **(58~59)** : 여기까지 딜러와 플레이어 모두 합이 22이 넘으면 안되므로 프로그래밍이 잘 되었는지 `assert`를 통해 검사한다.
* **(61)** : 게임을 시작한다.
* **(63~64)** : 플레이어부터 시작한다.
* **(65~67)** : `initial_action`이 `None`이 아니라면 해당 행동을 `action`으로 저장하고 `initial_action=None`을 통해 한번만 실행되도록 한다.
* **(68~70)** : `initial_action`이 `None`이라면(첫 행동 이후는 무조건 `initial_action`이 `None`이다.) 함수의 인자로 들어온 `policy_player`에 해당하는 정책을 통해 행동을 결정한다.
* **(72~73)** : 몬테 카를로 알고리즘이기 때문에 (+importance sampling을 위해) player의 trajectory를 저장한다. 이 이후에 카드를 뽑고 bust가 일어날 경우 함수가 바로 리턴되므로 trajectory의 `plyaer_sum`은 21이하이다.
* **(75~76)** : 행동이 스탠드일 경우(=카드를 더이상 받지 않을 경우) 플레이어의 턴을 종료한다.
* **(77~78)** : 행동이 히트일 경우 카드를 더 받는다.
* **(79~81)** : 에이스는 여러 개를 가질 수 있으므로 에이스의 개수를 세는 `ace_count`를 선언한다. 엄밀히는 1로 변환이 가능한 에이스의 개수이다. 해당 변수에 `usable_ace_player`이 `True`일 경우 1, `False`일 경우 0을 저장한다. 에이스는 처음에 두 장을 받고 22일 경우 한 장을 1로 만드므로 최대 1개이다.
* **(82~83)** : **(77~78)**에서 받은 카드가 1(에이스)일 경우 에이스의 개수(`ace_count`)를 1 늘린다.
* **(84)** : 카드의 합을 나타내는 `player_sum`에 카드의 번호를 누적한다.
* **(85~85)** : `player_sum`(플레이어가 가진 숫자의 합)이 21을 넘고 사용 가능한 에이스가 있다면 `player_sum`에서 10을 뺀다(에이스를 11에서 1로 바꾼다), 그리고 사용가능한 에이스의 양을 하나 줄인다.
* **(90~91)** : 플레이어가 가진 숫자의 합이 21을 넘어갈 경우 `return state, reward, trajectory` 을 통해 게임을 종료한다. (player busts)
  * player 패배 → reward = -1
* **(92)** : `player_sum` (플레이어가 가진 숫자의 합)이 22이상일 경우 `assert`를 호출한다(지금까지 함수가 진행되었다면 `player_sum`은 21이하여야 한다)
* **(93)** : `ace_count`가 1이라면 (2이상이면 11이 두개라는 뜻이므로 불가능하다) `useable_ace_player`를 `True`로 한다.
* **(95~96)** : 딜러의 차례를 진행한다
* **(97~98)** : 딜러 숫자의 합에따라 정해진 정책대로 행동을 선택한다
  * 16이하 : Hit, 17이상 : Stand
* **(99~100)** : 행동이 stand일 경우 딜러의 턴을 종료한다.
* **(101~102)** : 이후는 hit인 경우의 수이므로 새로운 카드를 뽑는다.
* **(103)** : 사용 가능한 에이스(11에서 1로 전환가능한 에이스)가 있을 경우 `ace_count`를 1로 지정한다. (2개는 22이기 때문에 불가능하다)
* **(104~105)** : 만약 새로 뽑은 카드가 1(에이스) 일 경우 `ace_count`를 1 증가시킨다
* **(106)** : 새로 뽑은 카드를 `dealer_sum`(딜러의 카드합)에 더한다. (에이스면 11을 더한다)
* **(107~110)** : 딜러가 가진 카드의 합이 21을 넘고 사용 가능한 에이스가 있을 경우 에이스 하나를 사용해 11을 1로 만들고 `ace_count`를 1 줄인다.
* **(111~113)** : 딜러 가진 숫자의 합이 21을 넘어갈 경우 `return state, reward, trajectory` 을 통해 게임을 종료한다. (dealer busts)
  * player 승리 → reward = +1
* **(114)** : `ace_count`가 1일 경우 (1개 아니면 0개이다) `usable_ace_dealer`를 `True`로 설정한다.
* **(116)** : 플레이어와 딜러가 모두 bust가 아닐 경우 카드의 합을 비교하여 승부를 결정한다.
* **(117)** : 여기까지 왔다면 플레이어와 딜러 모두 bust가 아니므로 둘다 카드의 합이 21이하여야 한다. 22이 이상이라면 코드의 오류가 있다는 것이므로 `assert`한다
* **(118~123)** : 플레이어와 딜러의 카드의 합을 비교하여 합이 더 높은 쪽이 이기고 결과를 reward에 반영하여 (state, reward, trajectory) 형태로 반환한다.
  * 플레이어 승리 : +1, 무승부 : 0, 플레이어 패배 : -1


## monte_carlo_on_policy
```python
# Monte Carlo Sample with On-Policy
def monte_carlo_on_policy(episodes):
    states_usable_ace = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    states_no_usable_ace_count = np.ones((10, 10))
    for i in tqdm(range(0, episodes)):
        _, reward, player_trajectory = play(target_policy_player)
        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            player_sum -= 12
            dealer_card -= 1
            if usable_ace:
                states_usable_ace_count[player_sum, dealer_card] += 1
                states_usable_ace[player_sum, dealer_card] += reward
            else:
                states_no_usable_ace_count[player_sum, dealer_card] += 1
                states_no_usable_ace[player_sum, dealer_card] += reward
    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count
```

* **(1~2)** : On-policy로 몬테카를로를 수행하는 함수
* **(3)** : 사용 가능한 에이스가 **있는** 경우 보상의 누적
* **(4~5)** : 사용 가능한 에이스가 **있는** 경우 각 상태의 방문 횟수, `division by zero`를 방지하기 위해 1로 시작한다.
* **(6)** : 사용 가능한 에이스가 **없는** 경우 보상의 누적
* **(7~8)** : 사용 가능한 에이스가 **없는** 경우 각 상태의 방문 횟수, `division by zero`를 방지하기 위해 1로 시작한다.
* **(9)** : 인수로 주어진 `episodes` 횟수만큼 에피소드를 실행한다.
* **(10)** : `play`를 통해 얻어진 정보를 저장한다 $\gamma=1$이므로 보상은 모든 상태에서 같다. `player_trajectory`에는 각 요소가 `[(usable_ace_player, player_sum, dealer_card1), action]` 형태로 저장되어 있다.
* **(11)** : 에피소드의 각 단계에 대해 반복수행, $\gamma=1$이므로 reward는 결과값 하나만 있으면 되고 상태가치 함수를 얻을 것이므로 행동($A$)은 사용하지 않는다. $0 < \gamma < 1$ 이면 $G \leftarrow \gamma G + R_{t+1}$을 이용하여 마지막부터 계산하는 것이 편하다. 
* **(12)** : 11이하는 상태에 있어서 사용될 일이 없다. 왜냐하면 11에서는 어떤 카드를 받아도 bust가 아니기 때문이다. 그러므로 이 이하는 승부를 결정하는 상태에서 고려될 필요가 없다. 그러므로 `player_sum`에서 12를 빼서 인덱스를 0부터 표현한다. [`play`](#play) **(72~73)** 에서 설명했다시피 `player_sum`은 무조건 21이하이므로 이는 0~9로 10개의 인덱스를 가진다.
* **(13)** : 딜러의 카드는 1~10인데 배열에 효율적으로 저장하기 위해 1을 뺀다. 나중에 플로팅할 때는 1~10으로 표현한다.
* **(14~16)** : 사용 가능한 에이스가 있는 상태일 경우 (`usable_ace == True`)
  * `states_usable_ace_count`를 1 증가
  * `states_usable_ace`에 reward를 더한다.
* **(17~19)** : 사용 가능한 에이스가 없는 상태일 경우 (`usable_ace == False`)
  * `states_no_usable_ace_count`를 1 증가
  * `states_no_usable_ace`에 reward를 더한다.
* **(20)** : 각 상태의 가치의 보상의 합을 상태의 방문횟수로 나누어 각 상태의 기댓값을 구한후 반환한다.