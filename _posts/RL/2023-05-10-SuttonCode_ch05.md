---
layout: single
title: "단단한 강화학습 코드 정리, chap5"
date: 2023-05-10 18:51:36
lastmod : 2023-05-10 18:51:36
categories: RL
tag: [Sutton, 단단한 강화학습, RL]
toc: true
toc_sticky: true
use_math: true
---


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
