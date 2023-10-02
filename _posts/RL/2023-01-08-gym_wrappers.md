---
layout: single
title: "gym Wrappers 정리"
date: 2023-01-08 13:58:15
lastmod : 2023-10-02 23:01:31
categories: RL
tag: [gymnaisum, gym, wrappers, RL]
toc: true
toc_sticky: true
---

`gym`이 `gymnasium`으로 바뀌었으나 서술의 편의를 위하여 `gym`으로 서술하겠다.

강화학습 학습시 `gym`이나 `wrappers`를 사용할 일이 참 많다. `Vector`와 함께 사용될때도 많은데 자주 사용하는 것 위주로 정리해본다. 자세한 설명은 [공식 문서](https://gymnasium.farama.org/)를 참고하기 바란다.

기본 사용 방법은 간단하다. 아래와 같이 환경을 Wrapper클래스 안에 인자로 넘겨주고 반환된 환경을 사용하면 된다. 기타 설정이 필요할 경우 파라미터도 같이 적어주면 된다.

사용시에 wrappers 간의 관계를 신경써야 한다. 예를 들어 `RecordEpisodeStatistics`를 사용하고 그것에 `RewardWrappers`를 사용하면 반영이 되지 않는다.

```python
env = gym.make(env_id)
env = gym.wrappers.YourWrapper(env, param1=param1, param2=param2, ...)
```

# Wrappers 우선순위

순위값이 높을수록 먼저 wrapping 해야 한다. 순위값은 임의로 설정한 것으로 공식적인 숫자가 아니다.

|순위|Wrappers|
|-|-|
|1|`Misc / Obs / Action / Reward Wrappers`|
|2|`TimeLimit`|
|3|`RecordVideo`|
|4|`RecordEpisodeStatistics`|
|5|`AutoResetWrapper`|

# RecordVideo

```python
class gymnasium.wrappers.RecordVideo(env: Env,
                                video_folder,
                                episode_trigger = None,
                                step_trigger = None,
                                video_length = 0,
                                name_prefix = 'rl-video',
                                disable_logger = False)
```

에피소드를 저장하는 wrapper이다. 결과 영상을 출력할 때나 디버그할 때 유용하다.

**Parameter**

* `env` : Wrappers를 적용할 환경
* `video_folder` : 영상을 저장할 폴더
* `episode_trigger` : 영상을 녹화할 조건이다. 200개의 episode마다 저장하고 싶다면 아래와 같이 하면 된다
  * `episode_trigger=lambda x: x % 200 == 0`
* `step_trigger` : 조건에 맞는 step 부터 에피소드가 끝날 때까지 녹화한다.
  * 예를 들어 어떤 에피소드가 384~712 스텝동안 진행되었다면 `step_trigger=lambda x: x % 200 == 0`일때 400~712 스텝이 녹화된다.
* `name_prefix` : 파일의 접두사이다. 귀찮아도 해주는게 편하다.
* `disable_logger` : `moviepy`의 logger의 출력여부를 결정한다. `False`로 하여 출력할 경우 아래와 같은 문구가 출력되는데 영상이 길거나 용량이 클 경우 진행상황을 알려준다. 보통 안하는 게 낫다.
```
Moviepy - Building video <PATH>
Moviepy - Writing video <PATH>
Moviepy - Done !
Moviepy - video ready <PATH>
```

**주의**

```python
# Gymnasium/gymnasium/wrappers/record_video.py
...

def _video_enabled(self):
    if self.step_trigger:
        return self.step_trigger(self.step_id)
    else:
        return self.episode_trigger(self.episode_id)

if not (self.terminated or self.truncated):
    # increment steps and episodes
    self.step_id += 1
    if not self.is_vector_env:
        if terminateds or truncateds:
            self.episode_id += 1
            self.terminated = terminateds
            self.truncated = truncateds
    elif terminateds[0] or truncateds[0]:
        self.episode_id += 1
        self.terminated = terminateds[0]
        self.truncated = truncateds[0]
...
```

코드를 보면 스텝에 따라 `step_id`가 증가하고 `terminated`, `truncated` 일 때 `episode_id`가 증가한다.

그러므로 `AutoResetWrapper`을 적용하기 전에 `RecordVideo`를 써야한다. `AutoResetWrapper`는 에피소드의 종료정보를 `info`를 통해 알려주기 때문에 `RecordVideo`에서는 그것에 대해 알 수가 없다.

## RecordVideoV0

여러 버그 수정과 기능 추가를 위하여 experimental에 새로운 RecordVideoV0 wrapper이 있다.

```python
class gymnasium.experimental.wrappers.RecordVideoV0(env,
                                    video_folder,
                                    episode_trigger = None,
                                    step_trigger = None,
                                    video_length = 0,
                                    name_prefix = 'rl-video',
                                    fps = None,
                                    disable_logger = False)
```

**추가된 Parameter**

* `fps` : 녹화된 영상의 fps를 설정한다. 설정하지 않을 경우 환경의 기본 metadata에 저장된 fps를 사용한다. 환경에 fps가 저장되어 있지 않을 경우 fps는 30이 적용된다.

확인한 차이는 다음과 같다.
1. 녹화시에 동영상의 meta파일이 저장되지 않는다.
   * 그냥 RecordVideo는 영상과 함께 메타 정보가 저장된다.
2. 전 에피소드의 마지막 프레임이 현재 에피소드에 같이 녹화되는 버그를 해결하였다.

# RecordEpisodeStatistics

```python
class gymnasium.wrappers.RecordEpisodeStatistics(env, deque_size = 100)
```

에피소드가 끝날 때마다 각 에피소드의 누적 보상, 에피소드 길이, 에피소드 경과 시간을 `info`에 반환한다. 즉 `info`가 빈 dict가 아니라면 `terminated=True` or `truncated=True`이다.

```
info
 └ 'episode'
     ├ 'r' : <array of cumulative reward>
     ├ 'l' : <array of episode length>
     └ 't' : <array of elapsed time since beginning of episode>
```

# AutoResetWrapper

```python
class gymnasium.wrappers.AutoResetWrapper(env)
```

`step` 호출시 `episode`가 끝날 때마다 자동으로 `reset`을 호출하는 Wrapper이다.

`env.step`이 호출될 때  `terminated=True` 또는 `truncated=True` 이 반환될 경우 `env.reset()`이 자동으로 호출된다. 이때 `env.step()`의 반환은 `(new_obs, final_reward, final_terminated, final_truncated, info)`이 된다. 여기서 `info`는 다음 내용을 반환한다. 만약 에피소드가 종료되지 않았다면(즉, `terminated=False` and `truncated=False`) `info`는 빈 dict를 반환한다.

```
info
 ├ 'final_observation' : final_observation
 └ 'final_info' : final_info
```

이때 `new_obs`, `final_observation`, `final_reward` ... 등등의 순서가 헷갈리는데 풀어쓰면 다음과 같다.

```python
    # termination == True
    final_observation, final_reward, final_terminated, final_truncated, info = env.step(action)
    if final_terminated or final_truncated:
        new_obs, _ = env.reset()
```

`AutoResetWrapper`, `RecordEpisodeStatistics` 두 개를 모두 적용하면 다음과 같이 된다. 에피소드가 종료하지 않았을 경우 빈 dict를 반환한다.

```python
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.AutoResetWrapper(env)
```

```
info
 ├ 'final_observation' : final_observation
 └ 'final_info'
    ├ 'episode'
    │   ├ 'r' : <array of cumulative reward>
    │   ├ 'l' : <array of episode length>
    │   └ 't' : <array of elapsed time since beginning of episode>
    └ ...
```

원리를 생각해보면 다음과 같다.
1. 에피소드 종료시 `RecordEpisodeStatistics`에 의해 `episode`키와 함께 에피소드 정보가 `info`에 등록된다.
2. `AutoResetWrapper` 는 에피소드 종료시 마지막 순간에 있었던 `info`를 `final_info` 안에 넣는다.

## RecordEpisodeStatistics와 AutoReset의 순서에 대해

왜 `RecordEpisodeStatistics`를 `AutoReset` 보다 먼저 적용하라는 것일까. 만약 환경이 한 개라면 상관이 없다. 

```python
env = gym.wrappers.AutoResetWrapper(env)
env = gym.wrappers.RecordEpisodeStatistics(env)
```
만약 위와 같이 `AutoReset`을 먼저 적용한다면 에피소드 종료시 `info`는 아래와 같을 것이다.

```text
info
 ├ 'final_observation' : final_observation
 ├ 'final_info' : final_info
 └ 'episode'
    ├ 'r' : <array of cumulative reward>
    ├ 'l' : <array of episode length>
    └ 't' : <array of elapsed time since beginning of episode>
```
1. 에피소드가 종료되었다면 `AutoReset`이 먼저 마지막으로 반환된 observation, info를 `info` 에 넣는다.
2. 그리고 `RecordEpisodeStatistics` 는 종료된 에피소드의 정보를 `info` 에 넣는다

그러므로 위 결과는 이치에 맞다.

**하지만 환경을 Vectorize 하면 어떻게 될까**

환경을 Vectorize 한다는 것은 같은 환경을 복사하여 각각의 독립된 환경을 병렬로 처리하는 것을 의미한다. 환경을 Vectorize 할 경우 이는 각각의 환경이 맨 마지막에 `AutoReset`이 적용된 것 처럼 기능한다. 그러니 이런 경우와 일관된 함수를 적용할 수 있게 `AutoReset`을 맨 마지막에 쓰는 것이 좋다고 하는 것이다.

# TimeLimit

```python
class gymnasium.wrappers.TimeLimit(env, max_episode_steps = None)
```

`max_episode_steps`를 초과하면 `truncated` 신호를 발생시킨다. `max_episodes_steps == None`일 경우 `env.spec.max_episode_steps` 값이 사용된다.

`AutoResetWrapper`보다 먼저 적용해야 한다. `TimeLimit` 은 `truncated` 신호를 보내고 `AutoResetWrapper`는 신호를 처리하기 때문에 신호를 먼저 발생시켜야 한다.

이 또한 우선순위에 주의해야 한다. `RecordEpisodeStatistics`보다 앞서 사용되어야 한다. 왜냐하면 `RecordEpisodeStatistics`는 `terminated == True or truncated == True`일때 episode의 통계를 보여주는데 `TimeLimit`이 더 뒤에 있으면 통계도 나오지 않은 `info` 상태로 `truncated` 신호가 발생되어 에피소드가 끝났는데도 불구하고 통계를 보여주지 않는다.

# FrameStack

Experimental 의 `FrameStackObservationV0`을 기준으로 설명한다. 엄밀히 말하면 ObservationStack이 맞는 표현같다.

```python
class gymnasium.experimental.wrappers.FrameStackObservationV0(env: gym.Env[ObsType, ActType], stack_size: int, *, zeros_obs: ObsType | None = None)
```

`stack_size` 개수만큼의 프레임을 겹쳐서 observation으로 반환한다.

반환되는 observation의 shape는 `(stack_size, )` + `(env.observation_space)` 가 된다.

에피소드 초반의 경우 앞 내용이 존재하지 않는데 이 부분은 0으로 채운다. 한 에피소드가 종료될 경우 새로운 에피소드의 observation은 이전 에피소드의 프레임이 아니라 0으로 채운다.

[`MountainCar-v0`](https://gymnasium.farama.org/environments/classic_control/mountain_car/)을 예로 들어보자 `MountainCar-v0`은 2개의 실수를 observation으로 반환한다.

```python
env = gym.make("MountainCar-v0")
env = gym.experimental.wrappers.FrameStackObservationV0(env, 5)
```

아래와 같이 observation_space가 변하는 것을 확인할 수 있다.

```python
>>> env = gym.make("MountainCar-v0")
>>> print(env.observation_space)
Box([-1.2  -0.07], [0.6  0.07], (2,), float32)

>>> env = gym.experimental.wrappers.FrameStackObservationV0(env, 5)
>>> print(env.observation_space)
Box([[-1.2  -0.07]
 [-1.2  -0.07]
 [-1.2  -0.07]
 [-1.2  -0.07]
 [-1.2  -0.07]], [[0.6  0.07]
 [0.6  0.07]
 [0.6  0.07]
 [0.6  0.07]
 [0.6  0.07]], (5, 2), float32)
```

아직 경험하지 않은 부분은 0으로 채우는 것을 확인할 수 있다.

```python
# obs, _ = env.reset(), step=1
[[ 0.          0.        ]
 [ 0.          0.        ]
 [ 0.          0.        ]
 [ 0.          0.        ]
 [-0.53346133  0.        ]]

# obs, _, _, _, _ = env.step(action), step=2
[[ 0.          0.        ]
 [ 0.          0.        ]
 [ 0.          0.        ]
 [-0.53346133  0.        ]
 [-0.53438735 -0.00092604]]
```

주의사항
* observation shape가 (1, 15, 15)일 때 5개의 Frame을 Stack하면 (5, 1, 15, 15)가 된다. `gymnasium.experimental.wrappers.ReshapeObservationV0` 등을 통해 (1, 15, 15)를 (15, 15)로 만들어줘야 한다.